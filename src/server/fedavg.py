import pickle
import sys
import os
import random
from pathlib import Path
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
import time
from typing import Dict, List, OrderedDict
from matplotlib import pyplot as plt

import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
from rich.console import Console
from rich.progress import track

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
from data.utils.datasets import DATASETS

from src.utils.tools import OUT_DIR, Logger, fix_random_seed, get_best_device, update_args_from_dict
from src.utils.models import get_model_arch
from src.client.fedavg import FedAvgClient


def local_time():
    now = int(round(time.time() * 1000))
    now02 = time.strftime(
        "%Y-%m-%d-%H:%M:%S", time.localtime(now / 1000)
    )  # e.g. 2023-11-08-10:31:47
    return now02


def get_fedavg_argparser():
    parser = ArgumentParser(description="FedAvg Server")

    # basic parameters
    parser.add_argument("-d", "--dataset", type=str, default="brats2019", help="dataset to use")
    parser.add_argument("-s", "--seed-server", type=int, default=42, help="random seed")
    parser.add_argument("--use-cuda", type=int, default=1)
    parser.add_argument(
        "--valid", "-v", default=True, type=bool, help="whether to use validation set"
    )
    # parameters for generation
    parser.add_argument(
        "--gen-model",
        type=str,
        default="cyclegan",
        help="Model architecture for the generation task.",
    )
    parser.add_argument(
        "--gen-round",
        type=int,
        default=5,
        help="Number of global rounds for the generation task.",
    )
    parser.add_argument(
        "--gen-epochs",
        type=int,
        default=2,
        help="Number of local training epochs for the generation model.",
    )
    parser.add_argument(
        "--gen-lr", type=float, default=0.0002, help="Learning rate for the generation model."
    )
    parser.add_argument(
        "--gen-batch-size", type=int, default=2, help="Batch size for the generation model."
    )
    parser.add_argument(
        "--gen-test-gap", "-gtg", type=int, default=1, help="The gap between two test rounds"
    )
    parser.add_argument("--gen-beta1", type=float, default=0.5, help="parameter for Adam")
    parser.add_argument("--gen-beta2", type=float, default=0.999, help="parameter for Adam")
    parser.add_argument(
        "--lambda-gan", type=float, default=1.0, help="parameter for the GAN loss function"
    )
    parser.add_argument(
        "--lambda-identity",
        type=float,
        default=10.0,
        help="parameter for the identity loss function",
    )
    parser.add_argument(
        "--lambda-cycle", type=float, default=10.0, help="parameter for the cycle loss function"
    )

    # paramters for classification
    parser.add_argument(
        "--class-model",
        type=str,
        default="resnet18",
        help="Model architecture for the classification task.",
    )
    parser.add_argument(
        "--class-epochs",
        type=int,
        default=30,
        help="Number of local training epochs for the classification model.",
    )
    parser.add_argument(
        "--class-round",
        type=float,
        default=0.001,
        help="Number of local training epochs for the classification model.",
    )
    parser.add_argument(
        "--class-lr", type=float, default=0.001, help="Learning rate for the classification model."
    )
    parser.add_argument("--save_log", type=int, default=1)

    return parser


class FedAvgServer:
    def __init__(
        self,
        algo: str = "FedAvg",
        args: Namespace = None,
    ):
        """
        load args & set random seed & create output directory & initialize logger
        """
        self.args = get_fedavg_argparser().parse_args() if args is None else args
        self.algo = algo
        fix_random_seed(self.args.seed_server)
        begin_time = str(local_time())
        self.out_dir = OUT_DIR / self.algo / self.args.dataset / begin_time
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        with open(
            PROJECT_DIR / "data" / self.args.dataset / "preprocessed_files" / "args.pkl", "rb"
        ) as f:
            self.args = update_args_from_dict(self.args, pickle.load(f))
        stdout = Console(log_path=False, log_time=False)
        logfile_path = os.path.join(self.out_dir, "log.html")
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.save_log,
            logfile_path=logfile_path,
        )
        self.logger.log(
            "=" * 20,
            "Generation Model:",
            self.args.gen_model,
            "Classification Model",
            self.args.class_model,
            "=" * 20,
        )
        self.logger.log("Experiment Arguments:", dict(self.args._get_kwargs()))
        self.test_result_list: List[Dict] = []  # store test result of each test round

        """
        initialize device, client, test dataset and validation dataset
        initialize generation model for test and validation
        """
        self.device = get_best_device(self.args.use_cuda)
        self.client = FedAvgClient(self.args, self.device, self.logger)
        # initialize test dataset and validation dataset
        self.transform = ToTensor()
        self.test_dataset = DATASETS[self.args.dataset](self.args, test_valid="test")
        test_loader = DataLoader(
            self.test_dataset, batch_size=round(self.args.gen_batch_size / 2), shuffle=True
        )
        self.gen_model_test = get_model_arch(self.args.gen_model)(
            self.args, self.device, test_loader, mode="test"
        )  # model for test
        if self.args.valid:
            self.valid_dataset = DATASETS[self.args.dataset](self.args, test_valid="valid")
            valid_loader = DataLoader(
                self.valid_dataset, batch_size=round(self.args.gen_batch_size / 2), shuffle=True
            )
            self.gen_model_valid = get_model_arch(self.args.gen_model)(
                self.args, self.device, valid_loader, mode="test"
            )

    def generation_workflow(self):
        # train generation model
        for round_idx in range(self.args.gen_round):
            self.logger.log("=" * 20, "Round:", round_idx, "Start Training", "=" * 20)
            for client_idx in range(self.args.client_num):
                self.client.train_gen_model(client_idx)
            client_weights_list = self.client.get_client_weights()
            model_dict_list_list = self.client.get_model_weights()
            aggregated_model_dict_list = self.gen_aggreation(
                model_dict_list_list, client_weights_list
            )
            self.client.download_model(aggregated_model_dict_list)
            # valid and test generation model
            if (round_idx + 1) % self.args.gen_test_gap == 0:
                self.gen_valid_test(aggregated_model_dict_list)
        self.gen_plot_test_result()
        self.generate_image()

    def generate_image(self):
        self.gen_model_test.move2device()
        self.gen_model_test.generate_image(os.path.join(self.out_dir, "generated_image"))
        self.gen_model_test.move2cpu()

    def gen_valid_test(self, aggregated_model_dict_list):
        if self.args.valid:
            self.gen_model_valid.download_weights(aggregated_model_dict_list)
            self.gen_model_valid.move2device()
            valid_result: Dict = self.gen_model_valid.test()
            self.gen_model_valid.move2cpu()
            self.logger.log("Validation Result")
            for key, value in valid_result.items():
                self.logger.log(f"{key}: {value:.4f}")
        self.gen_model_test.download_weights(aggregated_model_dict_list)
        self.gen_model_test.move2device()
        test_result: Dict = self.gen_model_test.test()
        self.test_result_list.append(test_result)
        self.gen_model_test.move2cpu()
        self.logger.log("Test Result")
        for key, value in test_result.items():
            self.logger.log(f"{key}: {value:.4f}")

    def gen_plot_test_result(self):
        path = os.path.join(self.out_dir, "metrics")
        if not os.path.exists(path):
            os.makedirs(path)
        test_rounds = range(self.args.gen_test_gap, self.args.gen_round + 1, self.args.gen_test_gap)
        cat_metric = {}
        for metric_name in self.test_result_list[0].keys():
            cat_metric[metric_name] = []
        for metric_at_round in self.test_result_list:
            for key, value in metric_at_round.items():
                cat_metric[key].append(value)
        for metric_name, value_list in cat_metric.items():
            plt.figure()
            plt.plot(test_rounds, value_list)
            plt.xlabel("Test Round")
            # plt.ylabel(metric_name)
            plt.ylabel("")
            plt.gca().ticklabel_format(useOffset=False, axis="y")
            plt.title(metric_name)
            plt.savefig(f"{path}/{metric_name}.png")

    def gen_aggreation(self, model_dict_list_list, client_weights_list) -> List[OrderedDict]:
        # aggregate model weights according to the weights of each client
        model_num = len(model_dict_list_list[0])
        aggregated_model_dict_list = [OrderedDict() for _ in range(model_num)]
        # 'model' here refers to models like generators in cyclegan, instead of the whole cyclegan, which is differnet with the 'model' in the FedAvgClient
        for client_idx, model_dict_list in enumerate(model_dict_list_list):
            for model_idx, model_dict in enumerate(model_dict_list):
                for key in model_dict.keys():
                    if client_idx == 0:
                        aggregated_model_dict_list[model_idx][key] = (
                            model_dict[key] * client_weights_list[client_idx]
                        )
                    else:
                        aggregated_model_dict_list[model_idx][key] += (
                            model_dict[key] * client_weights_list[client_idx]
                        )
        return aggregated_model_dict_list

    def classification(self):
        pass

    def run(self):
        self.generation()
        self.classification()

    def debug(self):
        self.gen_model_test.move2device()
        self.gen_model_test.generate_image(os.path.join(self.out_dir, "generated_image"))


if __name__ == "__main__":
    server = FedAvgServer()
    server.generation_workflow()
    # server.debug()
