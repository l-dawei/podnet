import copy
import logging
from mindspore import nn
import mindspore.ops as ops
from inclearn.lib.network.postprocessors import FactorScalar, HeatedUpScalar, InvertedFactorScalar
from inclearn.convnet.resnet import ResNet
from inclearn.lib import factory
from inclearn.lib.network.classifiers import CosineClassifier
from .word import Word2vec

logger = logging.getLogger(__name__)


# TODO: Need Unit Test
class BasicNet(nn.Cell):
    def __init__(
            self,
            convnet_type,
            convnet_kwargs={},
            classifier_kwargs={},
            postprocessor_kwargs={},
            wordembeddings_kwargs={},
            init="kaiming",
            device=None,
            return_features=False,
            extract_no_act=False,
            classifier_no_act=False,
            attention_hook=False,
            rotations_predictor=False,
            gradcam_hook=False
    ):
        print('111111111111111111111111111111111111111')
        super(BasicNet, self).__init__()

        """
        It seems that we only use FactorScalar of learned_scaling
        """
        if postprocessor_kwargs.get("type") == "learned_scaling":
            self.post_processor = FactorScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") == "inverted_learned_scaling":
            self.post_processor = InvertedFactorScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") == "heatedup":
            self.post_processor = HeatedUpScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") is None:
            self.post_processor = None
        else:
            raise NotImplementedError(
                "Unknown postprocessor {}.".format(postprocessor_kwargs["type"])
            )
        logger.info("Post processor is: {}".format(self.post_processor))

        self.convnet = factory.get_convnet(convnet_type, **convnet_kwargs)

        if "type" not in classifier_kwargs:
            raise ValueError("Specify a classifier!", classifier_kwargs)
        # if classifier_kwargs["type"] == "fc":
        #     self.classifier =
        # device need to convert to mindspore
        if classifier_kwargs["type"] == "cosine":
            logger.info('CosineClassifier input1:{}'.format(self.convnet.out_dim))
            logger.info('CosineClassifier input2:{}'.format(device))
            logger.info('CosineClassifier input3:{}'.format(classifier_kwargs))
            self.classifier = CosineClassifier(
                self.convnet.out_dim, device=device, **classifier_kwargs
            )
            logger.info('CosineClassifier output:{}'.format(self.classifier))
        else:
            raise ValueError("Unknown classifier type {}.".format(classifier_kwargs["type"]))

        # It seems that the rotations_predictor is always None
        if rotations_predictor:
            print("Using a rotations predictor.")
            self.rotations_predictor = nn.Dense(self.convnet.out_dim, 4)
        else:
            self.rotations_predictor = None

        # if wordembeddings_kwargs:
        #     self.word_embeddings = Word2vec(**wordembeddings_kwargs, device=device)
        # else:
        #     self.word_embeddings = None

        self.return_features = return_features
        self.extract_no_act = extract_no_act
        self.classifier_no_act = classifier_no_act
        self.attention_hook = attention_hook
        self.gradcam_hook = gradcam_hook
        self.device = device

        self.domain_classifier = None

        if self.gradcam_hook:
            self._hooks = [None, None]
            logger.info("Setting gradcam hook for gradients + activations of last conv.")
            self.set_gradcam_hook()
        if self.extract_no_act:
            logger.info("Features will be extracted without the last ReLU.")
        if self.classifier_no_act:
            logger.info("No ReLU will be applied on features before feeding the classifier.")


    def on_task_end(self):
        if isinstance(self.classifier, nn.Cell):
            self.classifier.on_task_end()
        if isinstance(self.post_processor, nn.Cell):
            self.post_processor.on_task_end()

    def on_epoch_end(self):
        if isinstance(self.classifier, nn.Cell):
            self.classifier.on_epoch_end()
        if isinstance(self.post_processor, nn.Cell):
            self.post_processor.on_epoch_end()

    def construct(self, x, rotation=False, index=None, features_processing=None, additional_features=None):
        words = None

        outputs = self.convnet(x)

        # if hasattr(self, "classifier_no_act") and self.classifier_no_act:
        #     selected_features = outputs[0]  # raw_features
        # else:
        #     selected_features = outputs[1]  # features

        # modelart
        selected_features = outputs[0]

        if features_processing is not None:
            selected_features = features_processing.fit_transform(selected_features)
        print('rotation:', rotation)
        if rotation:
            outputs["rotations"] = self.rotations_predictor(outputs["features"])
            nb_inputs = len(x) // 4
        else:
            if additional_features is not None:
                concat_op = ops.Concat(axis=0)
                clf_outputs = self.classifier(concat_op(selected_features, additional_features))
            else:
                clf_outputs = self.classifier(selected_features)

            """
            output is a list now instead of Pytorch is a dict
            output[0]:(raw_features)
            output[1]:(features)
            output[2]:(attentions)
            output[3]:(logits)
            output[4]:(raw_logits)
            """

            for i in clf_outputs:
                outputs.append(i)
            """
            Graph mode not support list append
            """
            # outputs.extend(clf_outputs)

            # outputs.update(clf_outputs)

        return outputs

    def post_process(self, x):
        if self.post_processor is None:
            return x
        return self.post_processor(x)

    @property
    def features_dim(self):
        return self.convnet.out_dim

    def add_classes(self, n_classes):
        self.classifier.add_classes(n_classes)

    def add_imprinted_classes(self, class_indexes, inc_dataset, **kwargs):
        # if hasattr(self.classifier, "add_imprinted_classes"):
        self.classifier.add_imprinted_classes(class_indexes, inc_dataset, self, **kwargs)

    def extract(self, x):
        outputs = self.convnet(x)
        if self.extract_no_act:
            # return outputs["raw_features"]
            return outputs[0]
        # return outputs["features"]
        return outputs[1]

    def freeze(self, trainable=False, model="all"):
        if model == "all":
            model = self
        elif model == "convnet":
            model = self.convnet
        elif model == "classifier":
            model = self.classifier
        else:
            assert False, model

        if not isinstance(model, nn.Cell):
            return self

        for param in model.trainable_params():
            param.requires_grad = trainable

        if not trainable:
            model.set_train(mode=False)
        else:
            model.set_train(mode=True)

        return self

    def get_group_parameters(self):
        groups = {"convnet": self.convnet.trainable_params()}

        if isinstance(self.post_processor, FactorScalar):
            groups["postprocessing"] = self.post_processor.trainable_params()
        # if hasattr(self.classifier, "new_weights"):
        groups["new_weights"] = self.classifier.new_weights
        # if hasattr(self.classifier, "old_weights"):
        groups["old_weights"] = self.classifier.old_weights
        # if hasattr(self.convnet, "last_block"):

        if isinstance(self.convnet, ResNet):
            groups["last_block"] = self.convnet.last_block.trainable_params()

        return groups

    def copy(self):
        return copy.deepcopy(self)

    @property
    def n_classes(self):
        return self.classifier.n_classes


"""
Unit Test
"""
# a = BasicNet(
#     'rebuffi',
#     convnet_kwargs={},
#     classifier_kwargs={'type': 'cosine', 'proxy_per_class': 10, 'distance': 'neg_stable_cosine_distance'},
#     postprocessor_kwargs={'type': 'learned_scaling', 'initial_value': 1.0},
#     device=[0],
#     return_features=True,
#     extract_no_act=True,
#     classifier_no_act=True,
#     attention_hook=True
# )
# x = factory.get_convnet('rebuffi')
# print(a)
