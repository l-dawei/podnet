import logging
from inclearn.models.base import IncrementalLearner
from inclearn.lib import factory, network, utils
import mindspore.ops as ops
import collections

EPSILON = 1e-8

logger = logging.getLogger(__name__)


class ICarl(IncrementalLearner):
    """
    Implementation of iCarl.
        # References:
            - iCaRL: Incremental Classifier and Representation Learning
            Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
            https://arxiv.org/abs/1611.07725
    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args):
        super().__init__()

        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._warmup_config = args.get("warmup", {})
        if self._warmup_config and self._warmup_config["total_epoch"] > 0:
            self._lr /= self._warmup_config["multiplier"]

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args["validation"]

        self._rotations_config = args.get("rotations_config", {})
        self._random_noise_config = args.get("random_noise_config", {})

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {
                "type": "fc",
                "use_bias": True
            }),
            device=self._device,
            extract_no_act=True,
            classifier_no_act=False,
            rotations_predictor=bool(self._rotations_config)
        )

        self._examplars = {}
        self._means = None
        self._herding_indexes = []
        self._data_memory, self._targets_memory = None, None

        self._old_model = None

        self._clf_loss = ops.BinaryCrossEntropy()
        self._distil_loss = ops.BinaryCrossEntropy()

        self._epoch_metrics = collections.defaultdict(list)

        self._meta_transfer = args.get("meta_transfer", {})

    def set_meta_transfer(self):
        if self._meta_transfer["type"] not in ("repeat", "once", "none"):
            raise ValueError(f"Invalid value for meta-transfer {self._meta_transfer}.")
