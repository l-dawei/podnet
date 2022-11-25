import copy
import logging
import collections
import math
import numpy as np
# import tqdm

from inclearn.lib import data, factory, network, utils, losses, herding
import mindspore as ms
from mindspore import Parameter, Tensor
from inclearn.models.base import IncrementalLearner
from mindspore.ops import stop_gradient
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Model

EPSILON = 1e-8

logger = logging.getLogger(__name__)


class PODNet(IncrementalLearner):
    """
    Pooled Output Distillation Network.
        # Reference:
            * Small Task Incremental Learning
              Douillard et al. 2020
    """

    def __init__(self, args):
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        # Optimization:
        self._batch_size = args["batch_size"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        # Rehearsal Learning:
        self._memory_size = args["memory_size"]
        self._fixed_memory = args.get("fixed_memory", True)
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args.get("validation")

        self._pod_flat_config = args.get("pod_flat", {})
        self._pod_spatial_config = args.get("pod_spatial", {})

        self._nca_config = args.get("nca", {})
        self._softmax_ce = args.get("softmax_ce", False)

        self._perceptual_features = args.get("perceptual_features")
        self._perceptual_style = args.get("perceptual_style")

        self._groupwise_factors = args.get("groupwise_factors", {})
        self._groupwise_factors_bis = args.get("groupwise_factors_bis", {})

        self._class_weights_config = args.get("class_weights_config", {})

        self._evaluation_type = args.get("eval_type", "icarl")
        self._evaluation_config = args.get("evaluation_config", {})

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._gradcam_distil = args.get("gradcam_distil", {})

        classifier_kwargs = args.get("classifier_config", {})
        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=classifier_kwargs,
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            device=self._device,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=args.get("classifier_no_act", True),
            attention_hook=True,
            gradcam_hook=bool(self._gradcam_distil)
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._finetuning_config = args.get("finetuning_config")

        self._herding_indexes = []

        self._weight_generation = args.get("weight_generation")

        self._meta_transfer = args.get("meta_transfer", {})

        # if self._meta_transfer:
        #     assert "mtl" in args["convnet"]

        self._post_processing_type = None
        self._data_memory, self._targets_memory = None, None

        self._args = args
        self._args["_logs"] = {}

        # TODO: List replace dict in Graph_mode
        # self._metrics = [0.0, 0.0, 0.0, 0.0]  # nca, flat, pod, loss
        # self._metrics = collections.defaultdict(float)
        self._metrics_nca = Parameter(Tensor(0.0, ms.float32), name="metrics_nca")
        self._metrics_flat = Parameter(Tensor(0.0, ms.float32), name="metrics_flat")
        self._metrics_pod = Parameter(Tensor(0.0, ms.float32), name="metrics_pod")
        self._metrics_loss = Parameter(Tensor(0.0, ms.float32), name="metrics_loss")

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    """
    Need Test
    """

    def _train_task(self, train_loader, val_loader):
        # TODO: hook smt
        # for p in self._network.trainable_paramters():
        #     if p.requires_grad:

        logger.debug("nb {}.".format(train_loader.get_dataset_size()))

        # _meta_transfer is None
        clipper = None

        # TODO: The most important part: the training part!!!
        # self._training_step(
        #     train_loader, val_loader, 0, , record_bn=True, clipper=clipper
        # )
        self._training_step(
            train_loader, self._optimizer, 0, self._n_epochs
        )

        self._post_processing_type = None

        if self._finetuning_config and self._task != 0:
            logger.info("Fine-tuning")

            if self._finetuning_config["sampling"] == "undersampling":
                self._data_memory, self._targets_memory, _, _ = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                loader = self.inc_dataset.get_memory_loader(*self.get_memory())

            if self._finetuning_config["tuning"] == "classifier":
                parameters = self._network.classifier.trainable_params()

            else:
                raise NotImplementedError(
                    "Unknwown finetuning parameters {}.".format(self._finetuning_config["tuning"])
                )

            # TODO: any other adjustment for ms
            # self._optimizer = factory.get_optimizer(
            #     parameters, self._opt_name, self._finetuning_config["lr"], self.weight_decay
            # )
            self._optimizer = nn.SGD(parameters, learning_rate=self._finetuning_config["lr"],
                                     weight_decay=self.weight_decay, momentum=0.9)

            self._scheduler = None

            # self._training_step(
            #     loader,
            #     val_loader,
            #     self._n_epochs,
            #     self._n_epochs + self._finetuning_config["epochs"],
            #     record_bn=False
            # )
            self._training_step(
                loader,
                self._optimizer, self._n_epochs,
                self._n_epochs + self._finetuning_config["epochs"]
            )

    @property
    def weight_decay(self):
        if isinstance(self._weight_decay, float):
            return self._weight_decay
        raise TypeError(
            "Invalid type {} for weight decay: {}.".format(
                type(self._weight_decay), self._weight_decay
            )
        )

    def _print_metrics(self, epoch, nb_epochs, nb_batches):
        # pretty_metrics = ", ".join(
        #     "{}: {}".format(metric_name, round(metric_value / nb_batches, 3))
        #     for metric_name, metric_value in self._metrics.items()
        # )

        pretty_metrics = ", ".join(
            "{}: {}".format('loss', round(self._metrics_loss / nb_batches, 3))
            # for metric_name, metric_value in self._metrics.items()
        )

        print(
            "T{}/{}, E{}/{} => {}".format(
                self._task + 1, self._n_tasks, epoch + 1, nb_epochs, pretty_metrics
            )
        )

    def _after_task_intensive(self, inc_dataset):
        self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
            inc_dataset, self._herding_indexes
        )

    def _after_task(self, inc_dataset):
        # self._old_model = self._network.copy().freeze().to(self._device) # TODO: to device
        self._old_model = self._network.copy().freeze()
        self._network.on_task_end()

    def _eval_task(self, test_loader):
        if self._evaluation_type in ("softmax", "cnn"):
            ypred = []
            ytrue = []

            softmax = ops.Softmax(axis=-1)

            for i, data in enumerate(test_loader.create_dict_iterator()):
                ytrue.append(data['label'].asnumpy())

                # TODO: to device?
                inputs = data["image"]
                # TODO: detach()
                logits = stop_gradient(self._network(inputs)[1])

                # TODO: ypred.append(preds.cpu().numpy()), so cpu() ?
                preds = softmax(logits)
                ypred.append(preds.asnumpy())

            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)

            self._last_results = (ypred, ytrue)

            return ypred, ytrue

        else:
            raise ValueError(self._evaluation_type)

    def _gen_weights(self):
        if self._weight_generation:
            utils.add_new_weights(
                self._network, self._weight_generation if self._task != 0 else "basic",
                self._n_classes, self._task_size, self.inc_dataset
            )

    def _before_task(self, train_loader, val_loader):
        self._gen_weights()
        self._n_classes += self._task_size
        logger.info("Now {} examplars per class.".format(self._memory_per_class))

        if self._groupwise_factors and isinstance(self._groupwise_factors, dict):
            if self._groupwise_factors_bis and self._task > 0:
                logger.info("Using second set of groupwise lr.")
                groupwise_factor = self._groupwise_factors_bis
            else:
                groupwise_factor = self._groupwise_factors

            params = []
            for group_name, group_params in self._network.get_group_parameters().items():
                if group_params is None or group_name == "last_block":
                    continue
                factor = groupwise_factor.get(group_name, 1.0)
                if factor == 0.:
                    continue
                params.append({"params": group_params, "lr": self._lr * factor})
                print(f"Group: {group_name}, lr: {self._lr * factor}.")

        else:
            params = self._network.trainable_params()

        # print(type(params[2]))
        # print(params[2])
        # TODO:
        params[2]['params'] = [params[2]['params']]
        params = [params[0], params[2]]
        # print(params)
        """
        In Ms, I firstly define scheduler, so that optimizer could use it
        """
        # self._scheduler = factory.get_lr_scheduler(self._scheduling, nb_epochs=self._n_epochs)
        self._scheduler = nn.CosineDecayLR(min_lr=.0, max_lr=0.1, decay_steps=self._n_epochs)

        # optimizer seems okay, its input is a list of para -> (params)
        # self._optimizer = factory.get_optimizer(params, self._opt_name, self._scheduler, self.weight_decay)
        # self._optimizer = nn.SGD([params[0]], learning_rate=self._scheduler, weight_decay=self.weight_decay,
        #                          momentum=0.9)
        self._optimizer = nn.SGD(params, learning_rate=self._scheduler, weight_decay=self.weight_decay,
                                 momentum=0.9)

        # self._class_weights_config is None
        self._class_weights = None

    def _compute_loss(self, inputs, outputs, targets, memory_flags):
        features, logits, atts = outputs[0], outputs[3], outputs[2]

        old_features = Tensor(np.zeros_like(features), ms.float32)
        old_atts = Tensor(np.zeros_like(atts), ms.float32)
        loss = 0.0

        # if self._post_processing_type is None:
        #     scaled_logits = self._network.post_process(logits)
        # else:
        #     scaled_logits = logits * self._post_processing_type

        if self._old_model is not None:
            # TODO: with torch.no_grad(): ?
            old_outputs = self._old_model(inputs)
            old_features = old_outputs[0]
            old_atts = old_outputs[2]

        if self._nca_config:
            nca_config = copy.deepcopy(self._nca_config)
            if self._network.post_processor:
                nca_config["scale"] = self._network.post_processor.factor

            loss = losses.nca(
                logits,
                targets,
                memory_flags=memory_flags,
                class_weights=self._class_weights,
                **nca_config
            )

            # self._metrics["nca"] += float(loss.asnumpy())
            self._metrics_nca += float(loss.asnumpy())

        # --------------------
        # Distillation losses:
        # --------------------

        if self._old_model is not None:
            if self._pod_flat_config:
                if self._pod_flat_config["scheduled_factor"]:
                    factor = self._pod_flat_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._pod_flat_config.get("factor", 1.)

                pod_flat_loss = factor * losses.embeddings_similarity(old_features, features)
                loss += pod_flat_loss

                # self._metrics["flat"] += float(pod_flat_loss.asnumpy())
                self._metrics_flat += float(pod_flat_loss.asnumpy())

            if self._pod_spatial_config:
                if self._pod_spatial_config.get("scheduled_factor", False):
                    factor = self._pod_spatial_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._pod_spatial_config.get("factor", 1.)

                # TODO: .bool() ?
                pod_spatial_loss = factor * losses.pod(
                    old_atts,
                    atts,
                    memory_flags=memory_flags.bool(),
                    task_percent=(self._task + 1) / self._n_tasks,
                    **self._pod_spatial_config
                )
                loss += pod_spatial_loss

                # self._metrics["pod"] += float(pod_spatial_loss.asnumpy())
                self._metrics_pod += float(pod_spatial_loss.asnumpy())

        return loss

    # ----------
    # Public API from ICarl
    # ----------
    # TODO: The most important part: the training part!!! unfinished
    def _training_step(
            self, train_loader, optim, initial_epoch, nb_epochs
    ):
        # best_epoch, best_acc = -1, -1.
        # wait = 0
        # grad, act = None, None

        # TODO: Need to fix out how to support multi device like torch did
        training_network = self._network

        # loss = ForwardLoss(self)

        # print(loss)

        loss_net = CustomWithLossCell(training_network, loss_fn=self._compute_loss)  # TODO: Test needed

        train_net = nn.TrainOneStepCell(loss_net, optim)

        # print(train_net)

        # model = Model(network=loss_net, optimizer=self._optimizer)
        #
        # model.train(epoch=nb_epochs, train_dataset=train_loader)

        for epoch in range(initial_epoch, nb_epochs):
            # self._metrics = collections.defaultdict(float)
            # self._metrics = {"nca": 0.0, "flat": 0.0, "loss": 0.0, "pod": 0.0}
            self._metrics_nca = Parameter(Tensor(0.0, ms.float32))
            self._metrics_flat = Parameter(Tensor(0.0, ms.float32))
            self._metrics_pod = Parameter(Tensor(0.0, ms.float32))
            self._metrics_loss = Parameter(Tensor(0.0, ms.float32))

            self._epoch_percent = epoch / (nb_epochs - initial_epoch)

            # prog_bar = tqdm(
            #     train_loader,
            #     disable=None,
            #     ascii=True,
            #     bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            # )

            for i, input_dict in enumerate(train_loader.create_dict_iterator(), start=1):
                inputs, targets = input_dict["image"], input_dict["label"]
                memory_flags = input_dict["memory_flag"]

                loss = train_net(inputs, targets, memory_flags)

                # self._metrics["loss"] += loss.asnumpy()
                self._metrics_loss += loss.asnumpy()

                print(loss, end='    ')
                # print(loss)

                self._print_metrics(epoch, nb_epochs, i)

        # if grad is not None:
        #     _clean_list(grad)
        #     _clean_list(act)

        # TODO:
        # loss = self._forward_loss(
        #     training_network,
        #     inputs,
        #     targets,
        #     memory_flags
        # )

    # def _forward_loss(
    #         self,
    #         training_network,
    #         inputs,
    #         targets,
    #         memory_flags,
    # ):
    #     outputs = training_network(inputs)
    #
    #     loss = self._compute_loss(inputs, outputs, targets, memory_flags)
    #
    #     if not utils.check_loss(loss):
    #         raise ValueError("A loss is NaN: {}".format(self._metrics))
    #
    #     self._metrics["loss"] += float(loss.asnumpy())
    #
    #     return loss

    # -----------------
    # Memory management
    # -----------------

    def build_examplars(
            self, inc_dataset, herding_indexes, memory_per_class=None, data_source="train"
    ):
        logger.info("Building & updating memory.")
        memory_per_class = memory_per_class or self._memory_per_class
        herding_indexes = copy.deepcopy(herding_indexes)

        data_memory, targets_memory = [], []
        class_means = np.zeros((self._n_classes, self._network.features_dim))

        for class_idx in range(self._n_classes):
            # We extract the features, both normal and flipped:
            inputs, loader = inc_dataset.get_custom_loader(
                class_idx, mode="test", data_source=data_source
            )
            features, targets = utils.extract_features(self._network, loader)
            features_flipped, _ = utils.extract_features(
                self._network,
                inc_dataset.get_custom_loader(class_idx, mode="flip", data_source=data_source)[1]
            )

            if class_idx >= self._n_classes - self._task_size:
                # New class, selecting the examplars:
                if self._herding_selection["type"] == "icarl":
                    selected_indexes = herding.icarl_selection(features, memory_per_class)

                else:
                    raise ValueError(
                        "Unknown herding selection {}.".format(self._herding_selection)
                    )

                herding_indexes.append(selected_indexes)

            # Reducing examplars:
            try:
                selected_indexes = herding_indexes[class_idx][:memory_per_class]
                herding_indexes[class_idx] = selected_indexes
            except:
                import pdb
                pdb.set_trace()

            # Re-computing the examplar mean (which may have changed due to the training):
            examplar_mean = self.compute_examplar_mean(
                features, features_flipped, selected_indexes, memory_per_class
            )

            data_memory.append(inputs[selected_indexes])
            targets_memory.append(targets[selected_indexes])

            class_means[class_idx, :] = examplar_mean

        data_memory = np.concatenate(data_memory)
        targets_memory = np.concatenate(targets_memory)

        return data_memory, targets_memory, herding_indexes, class_means

    def get_memory(self):
        return self._data_memory, self._targets_memory

    @staticmethod
    def compute_examplar_mean(feat_norm, feat_flip, indexes, nb_max):
        D = feat_norm.T
        D = D / (np.linalg.norm(D, axis=0) + EPSILON)

        D2 = feat_flip.T
        D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

        selected_d = D[..., indexes]
        selected_d2 = D2[..., indexes]

        mean = (np.mean(selected_d, axis=1) + np.mean(selected_d2, axis=1)) / 2
        mean /= (np.linalg.norm(mean) + EPSILON)

        return mean


"""
 Ms loss for train
"""


# def compute_all_loss():

# class ForwardLoss(nn.Cell):
#     def __init__(self, model):
#         """
#         :type model: PODNet
#         """
#         super(ForwardLoss, self).__init__()
#         self.model = model
#
#     def construct(self, inputs, outputs, targets, memory_flags):
#         """调用算子"""
#         features, logits, atts = outputs[0], outputs[3], outputs[2]
#
#         # self.model._metrics = collections.defaultdict(float)
#
#         if self.model._old_model is not None:
#             # TODO: with torch.no_grad(): ?
#             old_outputs = self.model._old_model(inputs)
#             old_features = old_outputs[0]
#             old_atts = old_outputs[2]
#
#         if self.model._nca_config:
#             nca_config = copy.deepcopy(self.model._nca_config)
#             if self.model._network.post_processor:
#                 nca_config["scale"] = self.model._network.post_processor.factor
#
#             loss = losses.nca(
#                 logits,
#                 targets,
#                 memory_flags=memory_flags,
#                 class_weights=self.model._class_weights,
#                 **nca_config
#             )
#             # self.model._metrics["nca"] += float(loss.asnumpy())
#
#         # --------------------
#         # Distillation losses:
#         # --------------------
#
#         if self.model._old_model is not None:
#             if self.model._pod_flat_config:
#                 if self.model._pod_flat_config["scheduled_factor"]:
#                     factor = self.model._pod_flat_config["scheduled_factor"] * math.sqrt(
#                         self.model._n_classes / self.model._task_size
#                     )
#                 else:
#                     factor = self.model._pod_flat_config.get("factor", 1.)
#
#                 pod_flat_loss = factor * losses.embeddings_similarity(old_features, features)
#                 loss += pod_flat_loss
#                 # self.model._metrics["flat"] += float(pod_flat_loss.asnumpy())
#
#             if self.model._pod_spatial_config:
#                 if self.model._pod_spatial_config.get("scheduled_factor", False):
#                     factor = self.model._pod_spatial_config["scheduled_factor"] * math.sqrt(
#                         self.model._n_classes / self.model._task_size
#                     )
#                 else:
#                     factor = self.model._pod_spatial_config.get("factor", 1.)
#
#                 # TODO: .bool() ?
#                 pod_spatial_loss = factor * losses.pod(
#                     old_atts,
#                     atts,
#                     memory_flags=memory_flags.bool(),
#                     task_percent=(self.model._task + 1) / self.model._n_tasks,
#                     **self.model._pod_spatial_config
#                 )
#                 loss += pod_spatial_loss
#                 # self.model._metrics["pod"] += float(pod_spatial_loss.asnumpy())
#
#         return loss


class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label, memory_flag):
        output = self._backbone(data)

        # self._metrics["loss"] += float(loss.asnumpy())

        return self._loss_fn(data, output, label, memory_flag)


def _clean_list(l):
    for i in range(len(l)):
        l[i] = None


"""
Unit Test
"""
# _network = network.BasicNet(
#     'rebuffi',
#     convnet_kwargs={},
#     classifier_kwargs={'type': 'cosine', 'proxy_per_class': 10, 'distance': 'neg_stable_cosine_distance'},
#     postprocessor_kwargs={'type': 'learned_scaling', 'initial_value': 1.0},
#     device=[0],
#     return_features=True,
#     extract_no_act=True,
#     classifier_no_act=True,
#     attention_hook=True,
#     gradcam_hook=False
# )
# params = []
# print(_network)
# for group_name, group_params in _network.get_group_parameters().items():
# print(group_name)
# print(group_params)
# print('--------------------')
# params.append({"params": group_params, "lr": 0.1})

# output:
# convnet
# [Parameter (name=convnet.conv_1_3x3.weight, shape=(16, 3, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.bn_1.gamma, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.bn_1.beta, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.0.conv_a.weight, shape=(16, 16, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.0.bn_a.gamma, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.0.bn_a.beta, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.0.conv_b.weight, shape=(16, 16, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.0.bn_b.gamma, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.0.bn_b.beta, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.1.conv_a.weight, shape=(16, 16, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.1.bn_a.gamma, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.1.bn_a.beta, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.1.conv_b.weight, shape=(16, 16, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.1.bn_b.gamma, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.1.bn_b.beta, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.2.conv_a.weight, shape=(16, 16, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.2.bn_a.gamma, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.2.bn_a.beta, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.2.conv_b.weight, shape=(16, 16, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.2.bn_b.gamma, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.2.bn_b.beta, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.3.conv_a.weight, shape=(16, 16, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.3.bn_a.gamma, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.3.bn_a.beta, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.3.conv_b.weight, shape=(16, 16, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.3.bn_b.gamma, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.3.bn_b.beta, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.4.conv_a.weight, shape=(16, 16, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.4.bn_a.gamma, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.4.bn_a.beta, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.4.conv_b.weight, shape=(16, 16, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.4.bn_b.gamma, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_1.blocks.4.bn_b.beta, shape=(16,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.0.conv_a.weight, shape=(32, 16, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.0.bn_a.gamma, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.0.bn_a.beta, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.0.conv_b.weight, shape=(32, 32, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.0.bn_b.gamma, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.0.bn_b.beta, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.1.conv_a.weight, shape=(32, 32, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.1.bn_a.gamma, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.1.bn_a.beta, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.1.conv_b.weight, shape=(32, 32, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.1.bn_b.gamma, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.1.bn_b.beta, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.2.conv_a.weight, shape=(32, 32, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.2.bn_a.gamma, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.2.bn_a.beta, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.2.conv_b.weight, shape=(32, 32, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.2.bn_b.gamma, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.2.bn_b.beta, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.3.conv_a.weight, shape=(32, 32, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.3.bn_a.gamma, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.3.bn_a.beta, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.3.conv_b.weight, shape=(32, 32, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.3.bn_b.gamma, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.3.bn_b.beta, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.4.conv_a.weight, shape=(32, 32, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.4.bn_a.gamma, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.4.bn_a.beta, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.4.conv_b.weight, shape=(32, 32, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.4.bn_b.gamma, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_2.blocks.4.bn_b.beta, shape=(32,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.0.conv_a.weight, shape=(64, 32, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.0.bn_a.gamma, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.0.bn_a.beta, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.0.conv_b.weight, shape=(64, 64, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.0.bn_b.gamma, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.0.bn_b.beta, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.1.conv_a.weight, shape=(64, 64, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.1.bn_a.gamma, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.1.bn_a.beta, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.1.conv_b.weight, shape=(64, 64, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.1.bn_b.gamma, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.1.bn_b.beta, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.2.conv_a.weight, shape=(64, 64, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.2.bn_a.gamma, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.2.bn_a.beta, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.2.conv_b.weight, shape=(64, 64, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.2.bn_b.gamma, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.2.bn_b.beta, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.3.conv_a.weight, shape=(64, 64, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.3.bn_a.gamma, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.3.bn_a.beta, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.3.conv_b.weight, shape=(64, 64, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.3.bn_b.gamma, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_3.blocks.3.bn_b.beta, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_4.conv_a.weight, shape=(64, 64, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_4.bn_a.gamma, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_4.bn_a.beta, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_4.conv_b.weight, shape=(64, 64, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_4.bn_b.gamma, shape=(64,), dtype=Float32, requires_grad=True), Parameter (name=convnet.stage_4.bn_b.beta, shape=(64,), dtype=Float32, requires_grad=True)]
# --------------------
# postprocessing
# [Parameter (name=post_processor.factor, shape=(), dtype=Float32, requires_grad=True)]
# --------------------
# print(params)
# import mindspore.nn as nn
#
# optim = nn.SGD(params, learning_rate=0.1, weight_decay=0.0)
# print(optim)
