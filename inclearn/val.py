import logging
import copy

import mindspore
from inclearn.models import PODNet
from inclearn.lib import logger as logger_lib
from inclearn.lib import utils, factory, results_utils, metrics
import os
import yaml
import json
import time
import sys
import statistics
import pickle
import random
import numpy as np

from inclearn.train import _set_results

logger = logging.getLogger(__name__)


def val(metric_logger,config, start_date,model,args, test_loader, run_id, task_id, task_info):
    model.load_parameters(config["resume"], run_id)
    ypreds, ytrue = model.eval_task(test_loader)
    print("eval_task done")

    metric_logger.log_task(
        ypreds, ytrue, task_size=task_info["increment"], zeroshot=None
    )
    if args["label"]:
        logger.info(args["label"])
    logger.info("Avg inc acc: {}.".format(metric_logger.last_results["incremental_accuracy"]))
    logger.info("Current acc: {}.".format(metric_logger.last_results["accuracy"]))
    logger.info(
        "Avg inc acc top5: {}.".format(metric_logger.last_results["incremental_accuracy_top5"])
    )
    logger.info("Current acc top5: {}.".format(metric_logger.last_results["accuracy_top5"]))
    logger.info("Forgetting: {}.".format(metric_logger.last_results["forgetting"]))
    logger.info("Cord metric: {:.2f}.".format(metric_logger.last_results["cord"]))
    if task_id > 0:
        logger.info(
            "Old accuracy: {:.2f}, mean: {:.2f}.".format(
                metric_logger.last_results["old_accuracy"],
                metric_logger.last_results["avg_old_accuracy"]
            )
        )
        logger.info(
            "New accuracy: {:.2f}, mean: {:.2f}.".format(
                metric_logger.last_results["new_accuracy"],
                metric_logger.last_results["avg_new_accuracy"]
            )
        )
    # if args.get("all_test_classes"):
    #     logger.info(
    #         "Seen classes: {:.2f}.".format(metric_logger.last_results["seen_classes_accuracy"])
    #     )
    #     logger.info(
    #         "unSeen classes: {:.2f}.".format(
    #             metric_logger.last_results["unseen_classes_accuracy"]
    #         )
    #     )
    results, results_folder = _set_results(args, start_date)
    results["results"].append(metric_logger.last_results)

    avg_inc_acc = results["results"][-1]["incremental_accuracy"]
    last_acc = results["results"][-1]["accuracy"]["total"]
    forgetting = results["results"][-1]["forgetting"]
    yield avg_inc_acc, last_acc, forgetting

    memory = model.get_memory()
    memory_val = model.get_val_memory()
