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

import torch.nn as nn

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
from data.utils.datasets import DATASETS

from src.utils.tools import (
    OUT_DIR,
    Logger,
    fix_random_seed,
    get_best_device,
    update_args_from_dict,
    move2cpu,
    move2device,
)
from src.model.model_tools import get_model_arch
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
    parser.add_argument("--multi-gpu", type=bool, default=True, help="whether to use multi-gpu")
    parser.add_argument("--save_log", type=int, default=1)
    parser.add_argument("--debug", default=True, type=bool, help="whether to debug the code")
    """
    # parameters for generation
    """
    parser.add_argument(
        "--gen-valid", "-gv", default=False, type=bool, help="whether to use validation set"
    )
    parser.add_argument(
        "--gen-model",
        type=str,
        default="cyclegan",
        help="Model architecture for the generation task.",
    )
    parser.add_argument(
        "--gen-round",
        type=int,
        default=60,
        help="Number of global rounds for the generation task.",
    )
    parser.add_argument(
        "--gen-epochs",
        type=int,
        default=1,
        help="Number of local training epochs for the generation model.",
    )
    parser.add_argument(
        "--gen-lr", type=float, default=0.0002, help="Learning rate for the generator."
    )
    parser.add_argument(
        "--dis-lr", type=float, default=0.0002, help="Learning rate for the discriminator"
    )
    parser.add_argument(
        "--gen-batch-size", type=int, default=3, help="Batch size for the generation model."
    )
    parser.add_argument(
        "--gen-test-gap", "-gtg", type=int, default=1, help="The gap between two test rounds"
    )
    parser.add_argument(
        "--upload-dis", type=bool, default=True, help="whether to upload discriminator weights"
    )

    parser.add_argument("--gen-beta1", type=float, default=0.5, help="parameter for Adam")
    parser.add_argument("--gen-beta2", type=float, default=0.999, help="parameter for Adam")
    parser.add_argument(
        "--lambda-gan", type=float, default=2.0, help="parameter for the GAN loss function"
    )
    parser.add_argument(
        "--lambda-identity",
        type=float,
        default=0.5,
        help="parameter for the identity loss function",
    )
    parser.add_argument(
        "--lambda-cycle", type=float, default=10.0, help="parameter for the cycle loss function"
    )
    parser.add_argument(
        "--lambda_paired", type=float, default=0.0, help="use paired loss for better generation"
    )
    parser.add_argument(
        "--gen-save-checkpoint",
        default=True,
        type=bool,
        help="whether to save the checkpoint of the trained generation model",
    )
    parser.add_argument(
        "--gen-resume",
        default="",
        type=str,
        help="the path of the checkpoint to resume the training of the generation model",
    )
    parser.add_argument("--dis-train-gap", default=1, type=int)
    """
    # paramters for classification
    """

    parser.add_argument(
        "--class-model",
        type=str,
        default="resnet50",
        help="Model architecture for the classification task.",
    )
    parser.add_argument(
        "--class-batchsize", type=int, default=2, help="Batch size for the classification model."
    )
    parser.add_argument(
        "--class-epochs",
        type=int,
        default=1,
        help="Number of local training epochs for the classification model.",
    )
    parser.add_argument(
        "--class-round",
        type=int,
        default=10,
        help="Number of local training epochs for the classification model.",
    )
    parser.add_argument(
        "--class-lr", type=float, default=0.001, help="Learning rate for the classification model."
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay for SGD")
    parser.add_argument(
        "--use-generator",
        type=bool,
        default=True,
        help="whether to use generator for classification",
    )
    parser.add_argument(
        "--class-test-gap", type=int, default=1, help="The gap between two test rounds"
    )
    parser.add_argument(
        "--class-valid",
        default=True,
        type=bool,
        help="whether to use validation set in classification task",
    )
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

        """
        Test and Validation
        """
        self.dataset_class = DATASETS[self.args.dataset]
        self.test_dataset = DATASETS[self.args.dataset](self.args, test_valid="test")
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=round(self.args.gen_batch_size / 2), shuffle=False
        )
        self.gen_model_test = get_model_arch(self.args.gen_model)(
            self.args, self.device, self.test_loader, mode="test"
        )
        self.class_model_test = get_model_arch(self.args.class_model)(self.args)
        if self.args.gen_valid or self.args.class_valid:
            self.valid_dataset = DATASETS[self.args.dataset](self.args, test_valid="valid")
            valid_loader = DataLoader(
                self.valid_dataset, batch_size=round(self.args.gen_batch_size / 2), shuffle=False
            )
            if self.args.gen_valid:
                self.gen_model_valid = get_model_arch(self.args.gen_model)(
                    self.args, self.device, valid_loader, mode="test"
                )

    def generation_workflow(self):
        # train generation model
        self.gen_round_begin = 0
        if self.args.gen_resume:
            self.gen_resume()
        for round_idx in range(self.gen_round_begin, self.args.gen_round):
            self.logger.log("=" * 20, "Round:", round_idx, "Generation Start Training", "=" * 20)
            for client_idx in range(self.args.client_num):
                self.client.gen_train_model(client_idx)

            client_weights_list = self.client.get_client_weights()
            model_dict_list_list = self.client.gen_get_model_weights()
            aggregated_model_dict_list = self.gen_aggreation(
                model_dict_list_list, client_weights_list
            )
            if self.args.gen_save_checkpoint:
                self.gen_save_checkpoint(round_idx, aggregated_model_dict_list)
            self.client.gen_download_model(aggregated_model_dict_list)
            # valid and test generation model
            if (round_idx + 1) % self.args.gen_test_gap == 0:
                self.gen_valid_test(aggregated_model_dict_list)
                self.generate_image()
        self.gen_plot_test_result()

    def generate_image(self):
        # generate some images from test set by the trained generation model
        self.gen_model_test.move2device()
        self.gen_model_test.generate_image(os.path.join(self.out_dir, "generated_image"))
        self.gen_model_test.move2cpu()

    def gen_valid_test(self, aggregated_model_dict_list):
        # process validation and test
        with torch.no_grad():
            if self.args.gen_valid:
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
        test_rounds = range(
            self.gen_round_begin + 1, self.args.gen_round + 1, self.args.gen_test_gap
        )
        self.gen_model_test.plot_test_result(self.test_result_list, test_rounds, path)

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

    def gen_save_checkpoint(self, round_idx, aggregated_model_dict_list):
        # save the trained generation model
        path = os.path.join(self.out_dir, "checkpoint")
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(
            {"round": round_idx, "model_state_dict_list": aggregated_model_dict_list},
            os.path.join(path, "checkpoint.pt"),
        )

    def gen_resume(self):
        checkpoint = torch.load(self.args.gen_resume)
        self.gen_round_begin = checkpoint["round"] + 1
        aggregated_model_dict_list = checkpoint["model_state_dict_list"]
        self.client.gen_download_model(aggregated_model_dict_list, resume=True)

    def classification_workflow(self):
        for round_idx in range(self.args.class_round):
            self.logger.log(
                "=" * 20, "Round:", round_idx, "Classification Start Training", "=" * 20
            )
            for client_idx in range(self.args.client_num):
                self.client.class_train_model(client_idx)

            client_weights_list = self.client.get_client_weights()
            model_dict_list_list = self.client.class_get_model_weights()
            aggregated_model_dict_list = self.gen_aggreation(
                model_dict_list_list, client_weights_list
            )
            self.client.class_download_model(aggregated_model_dict_list)
            if (round_idx + 1) % self.args.class_test_gap == 0:
                self.class_model_test.load_state_dict(aggregated_model_dict_list[0])
                self.class_valid_test()

        pass

    def class_valid_test(self):
        self.class_model_test = move2device(self.device, self.args.multi_gpu, self.class_model_test)
        self.class_model_test.eval()
        with torch.no_grad():
            if self.args.class_valid:
                loss = 0.0
                accurate_num = 0
                total_num = 0
                for batch in self.valid_loader:
                    image, label = self.dataset_class.class_organize_batch(
                        batch, self.args, self.device
                    )
                    output = self.class_model_test(image)
                    predictions = torch.argmax(output, dim=1)
                    accurate_num += (predictions == label).sum().item()
                    total_num += label.size(0)
                    loss += nn.functional.cross_entropy(output, label).item()
                self.logger.log(
                    "Validation Average Accuracy: {:.4f}".format(accurate_num / total_num)
                )
                self.logger.log(
                    "Validation Average Loss: {:.4f}".format(loss / len(self.valid_loader))
                )

            loss = 0.0
            accurate_num = 0
            total_num = 0
            for batch in self.test_loader:
                image, label = self.dataset_class.class_organize_batch(
                    batch, self.args, self.device
                )
                output = self.class_model_test(image)
                predictions = torch.argmax(output, dim=1)
                accurate_num += (predictions == label).sum().item()
                total_num += label.size(0)
                loss += nn.functional.cross_entropy(output, label).item()
            self.logger.log("Test Average Accuracy: {:.4f}".format(accurate_num / total_num))
            self.logger.log("Test Average Loss: {:.4f}".format(loss / len(self.test_loader)))
            self.class_model_test = move2cpu(self.class_model_test)

    def run(self):
        pass

    def debug(self):
        # self.args.multi_gpu = False
        self.gen_resume()
        self.classification_workflow()


if __name__ == "__main__":
    server = FedAvgServer()
    # server.args.debug = True
    if server.args.debug:
        server.debug()
    else:
        server.generation_workflow()
    # server.debug()
