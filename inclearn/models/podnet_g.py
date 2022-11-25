import copy
import logging
import collections
import math
import numpy as np
import time
from inclearn.lib import network, utils, losses, herding
from inclearn.models.base import IncrementalLearner

from mindspore.ops import stop_gradient
from mindspore.nn import LossBase

import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms
from mindspore import Parameter, Tensor

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

        self._device = 0

        self._batch_size = args["batch_size"]
        self._opt_name = 'sgd'
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        # TODO:
        # self._n_epochs = 1
        self._n_epochs = args["epochs"]

        self._scheduling = 'cosine'
        self._lr_decay = 0.1

        # Rehearsal Learning:
        self._memory_size = args["memory_size"]
        self._fixed_memory = args.get("fixed_memory", True)
        self._herding_selection = {'type': 'icarl'}
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = 0.0

        self._pod_flat_config = args.get("pod_flat")
        self._pod_spatial_config = args.get("pod_spatial")

        self._nca_config = {'margin': 0.6, 'scale': 1.0, 'exclude_pos_denominator': True}
        self._softmax_ce = False

        self._perceptual_features = None
        self._perceptual_style = None

        self._groupwise_factors = {'old_weights': 0.0}
        self._groupwise_factors_bis = {}

        self._class_weights_config = {}

        self._evaluation_type = 'cnn'
        self._evaluation_config = {}

        self._eval_every_x_epochs = None
        self._early_stopping = None

        self._gradcam_distil = None

        classifier_kwargs = {'type': 'cosine', 'proxy_per_class': 10, 'distance': 'neg_stable_cosine_distance'}

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=classifier_kwargs,
            postprocessor_kwargs={'type': 'learned_scaling', 'initial_value': 1.0},
            device=self._device,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=args.get("classifier_no_act", True),
            attention_hook=True,
            gradcam_hook=None
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._finetuning_config = args.get("finetuning_config")

        self._herding_indexes = []

        self._weight_generation = {'type': 'imprinted', 'multi_class_diff': 'kmeans'}

        self._meta_transfer = {}

        self._post_processing_type = None
        self._data_memory, self._targets_memory = None, None

        self._args = args
        self._args["_logs"] = {}

        # self._metrics_nca = Parameter(Tensor(0.0, ms.float32), name="metrics_nca")
        # self._metrics_flat = Parameter(Tensor(0.0, ms.float32), name="metrics_flat")
        # self._metrics_pod = Parameter(Tensor(0.0, ms.float32), name="metrics_pod")
        # self._metrics_loss = Parameter(Tensor(0.0, ms.float32), name="metrics_loss")

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    def _train_task(self, train_loader, val_loader):
        logger.debug("nb {}.".format(train_loader.get_dataset_size()))

        self._training_step(
            train_loader, self._optimizer, 0, self._n_epochs
        )

        self._post_processing_type = None
        print('aaaaaaaaaaaaaaaaaa', self._network)
        print('aaaaaaaaaaaaaaaaaa', self._network.classifier)
        # self._network.classifier.
        print('aaaaaaaaaaaaaaaaaa', self._network.classifier)
        print('aaaaaaaaaaaaaaaaaa', self._network.classifier.trainable_params())
        print('bbbbbbbbbbbbbbbbbb', self._finetuning_config["lr"])
        print('cccccccccccccccccc', self.weight_decay)
        print('--------------------------')
        if self._task != 0:
            logger.info("Fine-tuning")

            self._data_memory, self._targets_memory, _, _ = self.build_examplars(
                self.inc_dataset, self._herding_indexes
            )
            loader = self.inc_dataset.get_memory_loader(*self.get_memory())

            parameters = self._network.classifier.trainable_params()

            self._optimizer = nn.SGD(parameters, learning_rate=self._finetuning_config["lr"],
                                     weight_decay=self.weight_decay, momentum=0.9)

            self._scheduler = None

            self._training_step(
                loader,
                self._optimizer, self._n_epochs,
                self._n_epochs + self._finetuning_config["epochs"]
            )

    @property
    def weight_decay(self):
        if isinstance(self._weight_decay, float):
            return self._weight_decay
        elif isinstance(self._weight_decay, dict):
            start, end = self._weight_decay["start"], self._weight_decay["end"]
            step = (max(start, end) - min(start, end)) / (self._n_tasks - 1)
            factor = -1 if start > end else 1

            return start + factor * self._task * step
        raise TypeError(
            "Invalid type {} for weight decay: {}.".format(
                type(self._weight_decay), self._weight_decay
            )
        )
        return self._weight_decay

    # def _print_metrics(self, epoch, nb_epochs, nb_batches):
    #
    #     print(
    #         "T{}/{}, E{}/{} => {}".format(
    #             self._task + 1, self._n_tasks, epoch + 1, nb_epochs, pretty_metrics
    #         )
    #     )

    def _after_task_intensive(self, inc_dataset):
        self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
            inc_dataset, self._herding_indexes
        )

    def _after_task(self, inc_dataset):
        self._old_model = self._network.copy().freeze()
        self._network.on_task_end()

    def _eval_task(self, test_loader):
        if self._evaluation_type in ("softmax", "cnn"):
            ypred = []
            ytrue = []

            softmax = ops.Softmax(axis=-1)

            for i, data in enumerate(test_loader.create_dict_iterator()):
                logger.info('test_loader:{}.'.format(type(test_loader)))
                # logger.info('test_loader.shape:{}.'.format(test_loader.shape))
                test_dict=test_loader.create_dict_iterator()
                logger.info('test_dict:{}.'.format(type(test_dict)))
                # logger.info('test_dict.shape:{}.'.format(test_dict.shape))
                ytrue.append(data['label'].asnumpy())

                inputs = data["image"]
                logger.info('inputs_type:{}.'.format(type(inputs)))
                logger.info('inputs_shape:{}.'.format(inputs.shape))
                logger.info('_network(inputs):{}.'.format(len(self._network(inputs))))
                logger.info('_network(inputs):{}.'.format(type(self._network(inputs))))
                logger.info('_network(inputs)_item0:{}.'.format(self._network(inputs)[0].shape))
                logger.info('_network(inputs)_item1:{}.'.format(self._network(inputs)[1].shape))
                logger.info('_network(inputs)_item2:{}.'.format(type(self._network(inputs)[2])))
                logger.info('_network(inputs)_item2:{}.'.format(len(self._network(inputs)[2])))
                # logger.info('_network(inputs)_item3:{}.'.format(self._network(inputs)[3].shape))
                logger.info('_network(inputs)_item3:{}.'.format(type(self._network(inputs)[3])))
                logger.info('_network(inputs)_item3:{}.'.format(self._network(inputs)[3].shape))
                logger.info('_network(inputs)_item4:{}.'.format(self._network(inputs)[4].shape))
                logger.info('_network(inputs)_item4:{}.'.format(type(self._network(inputs)[4])))

                logits = stop_gradient(self._network(inputs)[1])
                logger.info('logits:{}.'.format(logits.shape))
                preds = softmax(logits)
                logger.info("Eval output shape size: {}.".format(preds.shape))
                ypred.append(preds.asnumpy())

            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)

            self._last_results = (ypred, ytrue)
            logger.info('ypred:{}.'.format(ypred.shape))
            logger.info('ytrue:{}.'.format(ytrue.shape))

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

        params[2]['params'] = [params[2]['params']]
        params = [params[0], params[2]]

        """
        In Ms, I firstly define scheduler, so that optimizer could use it
        """
        self._scheduler = nn.CosineDecayLR(min_lr=0.0, max_lr=self._lr, decay_steps=self._n_epochs)
        self._optimizer = nn.SGD(params, learning_rate=self._scheduler, weight_decay=self.weight_decay,
                                 momentum=0.9)
        self._class_weights = None

    # def _compute_loss(self, inputs, outputs, targets, memory_flags):
    #
    #     features, logits, atts = outputs[0], outputs[3], outputs[2]
    #
    #     old_features = Tensor(np.zeros_like(features), ms.float32)
    #     old_atts = Tensor(np.zeros_like(atts), ms.float32)
    #
    #     loss = 0.0
    #
    #     if self._old_model is not None:
    #         old_outputs = self._old_model(inputs)
    #         old_features = old_outputs[0]
    #         old_atts = old_outputs[2]
    #
    #     if self._nca_config:
    #         nca_config = copy.deepcopy(self._nca_config)
    #
    #         # TODO:
    #         if self._network.post_processor:
    #             nca_config["scale"] = self._network.post_processor.factor
    #
    #         loss = losses.nca(
    #             logits,
    #             targets,
    #             memory_flags=memory_flags,
    #             class_weights=self._class_weights,
    #             **nca_config
    #         )
    #
    #         self._metrics_nca += float(loss.asnumpy())
    #
    #     # --------------------
    #     # Distillation losses:
    #     # --------------------
    #
    #     if self._old_model is not None:
    #         if self._pod_flat_config:
    #             if self._pod_flat_config["scheduled_factor"]:
    #                 factor = self._pod_flat_config["scheduled_factor"] * math.sqrt(
    #                     self._n_classes / self._task_size
    #                 )
    #             else:
    #                 factor = self._pod_flat_config.get("factor", 1.)
    #
    #             pod_flat_loss = factor * losses.embeddings_similarity(old_features, features)
    #             loss += pod_flat_loss
    #
    #             self._metrics_flat += float(pod_flat_loss.asnumpy())
    #
    #         if self._pod_spatial_config:
    #             if self._pod_spatial_config.get("scheduled_factor", False):
    #                 factor = self._pod_spatial_config["scheduled_factor"] * math.sqrt(
    #                     self._n_classes / self._task_size
    #                 )
    #             else:
    #                 factor = self._pod_spatial_config.get("factor", 1.)
    #
    #             # TODO: .bool() ?
    #             pod_spatial_loss = factor * losses.pod(
    #                 old_atts,
    #                 atts,
    #                 memory_flags=memory_flags.bool(),
    #                 task_percent=(self._task + 1) / self._n_tasks,
    #                 **self._pod_spatial_config
    #             )
    #             loss += pod_spatial_loss
    #
    #             self._metrics_pod += float(pod_spatial_loss.asnumpy())
    #
    #     return loss

    # ----------
    # Public API from ICarl
    # ----------
    def _training_step(
            self, train_loader, optim, initial_epoch, nb_epochs
    ):
        training_network = self._network

        # TODO:
        com_loss = ComputeLoss(self._old_model, self._pod_flat_config,
                               self._n_classes, self._task_size, self._pod_spatial_config, self._class_weights,
                               self._task, self._n_tasks)

        loss_net = CustomWithLossCell(training_network, loss_fn=com_loss)  # TODO: Test needed

        train_net = nn.TrainOneStepCell(loss_net, optim)

        for epoch in range(initial_epoch, nb_epochs):
            # In Graph, Only para is allowed to modify
            # self._metrics_nca = Parameter(Tensor(0.0, ms.float32))
            # self._metrics_flat = Parameter(Tensor(0.0, ms.float32))
            # self._metrics_pod = Parameter(Tensor(0.0, ms.float32))
            # self._metrics_loss = Parameter(Tensor(0.0, ms.float32))

            # self._epoch_percent = epoch / (nb_epochs - initial_epoch)

            for i, input_dict in enumerate(train_loader.create_dict_iterator(), start=1):
                step_start_time = time.time()
                # print("before data get {}".format(i))
                inputs, targets = input_dict["image"], input_dict["label"]
                memory_flags = input_dict["memory_flag"]
                print('memory_flags:', memory_flags)
                print('type',type(memory_flags))
                # print("after data get {}".format(i))
                # loss = train_net(inputs, targets)
                loss = train_net(inputs, targets,memory_flags)

                # self._metrics_loss += loss.asnumpy()
                #
                print("T{}/{}, E{}/{}, step: {}, loss: {}".format(self._task + 1, self._n_tasks, epoch + 1,
                                                                  nb_epochs, i, loss))
                logger.info("step train spend {}s.".format(int(time.time() - step_start_time)))

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

        # TODO: Only for 25 most, more will die
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
            # try:
            #     selected_indexes = herding_indexes[class_idx][:memory_per_class]
            #     herding_indexes[class_idx] = selected_indexes
            #
            # except:
            #     import pdb
            #     pdb.set_trace()
            selected_indexes = herding_indexes[class_idx][:memory_per_class]
            herding_indexes[class_idx] = selected_indexes

            logger.info("compute_examplar_mean")

            # Re-computing the examplar mean (which may have changed due to the training):
            examplar_mean = self.compute_examplar_mean(
                features, features_flipped, selected_indexes, memory_per_class
            )

            data_memory.append(inputs[selected_indexes])
            targets_memory.append(targets[selected_indexes])

            class_means[class_idx, :] = examplar_mean

            logger.info("Building & updating memory for class {} done".format(class_idx))

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


class ComputeLoss(nn.Cell):
    def __init__(self, _old_model, _pod_flat_config, _n_classes,
                 _task_size, _pod_spatial_config, _class_weights, _task, _n_tasks):
        super(ComputeLoss, self).__init__()

        self._old_model = _old_model
        self._pod_flat_config = _pod_flat_config
        self._n_classes = _n_classes
        self._task_size = _task_size
        self._pod_spatial_config = _pod_spatial_config
        self._class_weights = _class_weights
        self._task = _task
        self._n_tasks = _n_tasks

        self.cl_ta = Tensor(math.sqrt(self._n_classes / self._task_size))

    def construct(self, inputs, outputs, targets, memory_flags):
        features, logits, atts = outputs[0], outputs[3], outputs[2]

        # old_features = Tensor(np.zeros_like(features), ms.float32)
        # old_atts = Tensor(np.zeros_like(atts), ms.float32)
        # old_features = ms.numpy.zeros_like(features, ms.float32)
        # old_atts = ms.numpy.zeros_like(atts, ms.float32)
        old_features = None
        old_atts = None
        loss = 0.0

        if self._old_model:
            old_outputs = self.old_model(inputs)
            old_features = old_outputs[0]
            old_atts = old_outputs[2]

            # if self._network.post_processor:
            #     nca_config["scale"] = self._network.post_processor.factor

        # print(logits.shape, type(logits))
        # print(targets.shape, type(targets))

        loss = losses.nca(logits, targets, self._class_weights)

        # self._metrics_nca += float(loss.asnumpy())

        # --------------------
        # Distillation losses:
        # --------------------

        factor = 0.0

        if self._old_model:
            # factor = self._pod_flat_config["scheduled_factor"] * math.sqrt(tmp)
            factor = self._pod_flat_config["scheduled_factor"] * self.cl_ta

            pod_flat_loss = factor * losses.embeddings_similarity(old_features, features)
            loss += pod_flat_loss

            # self._metrics_flat += float(pod_flat_loss.asnumpy())

            # factor = self._pod_spatial_config["scheduled_factor"] * math.sqrt(tmp)
            factor = self._pod_spatial_config["scheduled_factor"] * self.cl_ta

            pod_spatial_loss = factor * losses.pod(
                old_atts,
                atts,
                memory_flags=ops.cast(memory_flags,ms.bool_),
                task_percent=(self._task + 1) / self._n_tasks,
                **self._pod_spatial_config
            )
            loss += pod_spatial_loss

        # self._metrics_pod += float(pod_spatial_loss.asnumpy())

        return loss


class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label, memory_flag):
        output = self._backbone(data)

        return self._loss_fn(data, output, label, memory_flag)


def _clean_list(l):
    for i in range(len(l)):
        l[i] = None
