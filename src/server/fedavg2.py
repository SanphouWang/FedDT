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
from src.client.fedavg2 import FedAvgClient


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
    parser.add_argument("--debug", default=False, type=bool, help="whether to debug the code")
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
        default=50,
        help="Number of global rounds for the generation task.",
    )
    parser.add_argument(
        "--gen-samelr-round",
        type=int,
        default=20,
        help="Number of rounds with the same learning rate",
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
        "--gen-test-gap", "-gtg", type=int, default=2, help="The gap between two test rounds"
    )
    parser.add_argument(
        "--upload-dis", type=bool, default=True, help="whether to upload discriminator weights"
    )
    parser.add_argument(
        "--gen-eval-train",
        default=True,
        type=bool,
        help="whether to evaluate the generation model on the training set",
    )
    parser.add_argument("--gen-beta1", type=float, default=0.5, help="parameter for Adam")
    parser.add_argument("--gen-beta2", type=float, default=0.999, help="parameter for Adam")
    parser.add_argument(
        "--lambda-gan", type=float, default=1, help="parameter for the GAN loss function"
    )
    parser.add_argument(
        "--lambda-identity",
        type=float,
        default=10.0,
        help="parameter for the identity loss function",
    )
    parser.add_argument(
        "--lambda-cycle", type=float, default=100.0, help="parameter for the cycle loss function"
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
    parser.add_argument(
        "--only-resume-model",
        default=True,
        type=bool,
        help="whether to only resume the model weights",
    )
    """
    # paramters for classification
    """

    parser.add_argument(
        "--class-model",
        type=str,
        default="resnet_mixed_conv",
        help="Model architecture for the classification task.",
    )
    parser.add_argument(
        "--class-batch-size", type=int, default=5, help="Batch size for the classification model."
    )
    parser.add_argument(
        "--class-epochs",
        type=int,
        default=5,
        help="Number of local training epochs for the classification model.",
    )
    parser.add_argument(
        "--class-round",
        type=int,
        default=10,
        help="Number of local training epochs for the classification model.",
    )
    parser.add_argument(
        "--class-lr",
        type=float,
        default=0.0001,
        help="Learning rate for the classification model.",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay for SGD")
    parser.add_argument(
        "--use-generator",
        type=bool,
        default=False,
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
        if self.args.gen_resume == "" and self.args.use_generator:
            raise ValueError(
                "Please provide the path of the checkpoint if you want to use generator in classification task"
            )
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
        Initialize Dataset and Client
        """
        # Dataset
        self.generation_dataset_train = [
            DATASETS[self.args.dataset](self.args, client_id=i) for i in range(self.args.client_num)
        ]
        self.generation_dataset_test = DATASETS[self.args.dataset](self.args, client_id="test")
        self.generation_dataloader_test = DataLoader(
            self.generation_dataset_test, batch_size=self.args.gen_batch_size, shuffle=False
        )
        self.classification_dataset_train = [
            DATASETS[self.args.dataset](self.args, client_id=i) for i in range(self.args.client_num)
        ]
        self.classification_dataset_test = DATASETS[self.args.dataset](self.args, client_id="test")
        self.classification_dataloader_test = DataLoader(
            self.classification_dataset_test, batch_size=self.args.class_batch_size, shuffle=False
        )
        if self.args.gen_valid:
            self.generation_dataset_valid = DATASETS[self.args.dataset](
                self.args, client_id="valid"
            )
            self.generation_dataloader_valid = DataLoader(
                self.generation_dataset_valid, batch_size=self.args.gen_batch_size, shuffle=False
            )
        if self.args.class_valid:
            self.classification_dataset_valid = DATASETS[self.args.dataset](
                self.args, client_id="valid"
            )
            self.classification_dataloader_valid = DataLoader(
                self.classification_dataset_valid,
                batch_size=self.args.class_batch_size,
                shuffle=False,
            )
        # Client & Model
        self.client_list = [
            FedAvgClient(self.args, self.generation_dataset_train[i], self.logger)
            for i in range(self.args.client_num)
        ]
        self.device = get_best_device(self.args.use_cuda)
        self.generation_model = get_model_arch(self.args.gen_model)(
            self.args, self.logger, self.device
        )  # model for test
        self.classification_model = get_model_arch(self.args.class_model)()

    def generation_workflow(self):
        # train generation model
        self.gen_round_begin = 0
        if self.args.gen_resume:
            self.gen_resume()
        # calculate weights for each client for model aggregation
        client_datanum_list = [len(dataset) for dataset in self.generation_dataset_train]
        total_datanum = sum(client_datanum_list)
        client_weights_list = [datanum / total_datanum for datanum in client_datanum_list]
        for round_idx in range(self.gen_round_begin, self.args.gen_round):
            client_generation_model_weights_list: List[List[OrderedDict]] = []
            self.logger.log("=" * 20, "Round:", round_idx, "Generation Start Training", "=" * 20)
            # train generation model on each client
            for client_idx in range(self.args.client_num):
                self.logger.log(local_time(), " Client", client_idx, "Training Start")
                self.client_list[client_idx].generation_workflow(round_idx)
                client_generation_model_weights_list.append(
                    self.client_list[client_idx].get_generation_model_weights()
                )

            aggregated_model_weights = self.gen_aggreation(
                client_generation_model_weights_list, client_weights_list
            )

            for client_idx in range(self.args.client_num):
                self.client_list[client_idx].load_generation_model_weights(aggregated_model_weights)
            if self.args.gen_save_checkpoint:
                self.gen_save_checkpoint(round_idx, aggregated_model_weights)

            # valid and test generation model
            if (round_idx + 1) % self.args.gen_test_gap == 0:
                self.gen_valid_test(aggregated_model_weights)
                self.generate_image()

        self.gen_plot_test_result()

    def generate_image(self):
        # generate some images from test set by the trained generation model
        self.generation_model.move2device()
        self.generation_model.generate_image(
            os.path.join(self.out_dir, "generated_image"), self.generation_dataloader_test
        )
        self.generation_model.move2cpu()

    def gen_valid_test(self, aggregated_model_dict_list):
        # process validation and test
        with torch.no_grad():
            self.generation_model.download_weights(aggregated_model_dict_list)
            self.generation_model.move2device()
            if self.args.gen_valid:
                valid_result: Dict = self.generation_model.test(self.generation_dataloader_valid)
                self.logger.log("Validation Result")
                for key, value in valid_result.items():
                    self.logger.log(f"{key}: {value:.4f}")
            test_result: Dict = self.generation_model.test(self.generation_dataloader_test)
            self.test_result_list.append(test_result)
            self.generation_model.move2cpu()
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
        self.generation_model.plot_test_result(self.test_result_list, test_rounds, path)

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
        if not self.args.only_resume_model:
            self.gen_round_begin = checkpoint["round"] + 1
        aggregated_model_dict_list = checkpoint["model_state_dict_list"]
        for client in self.client_list:
            client.load_generation_model_weights(aggregated_model_dict_list, resume=True)

    def classification_workflow(self):
        client_datanum_list = [len(dataset) for dataset in self.classification_dataset_train]
        total_datanum = sum(client_datanum_list)
        client_weights_list = [datanum / total_datanum for datanum in client_datanum_list]
        for round_idx in range(self.args.class_round):
            client_classification_model_weights_list: List[List[OrderedDict]] = []
            self.logger.log(
                "=" * 20, "Round:", round_idx, "Classification Start Training", "=" * 20
            )
            for client_idx in range(self.args.client_num):
                self.logger.log("Client", client_idx, "Begin")
                self.client_list[client_idx].classification_workflow()
                client_classification_model_weights_list.append(
                    self.client_list[client_idx].get_classification_model_weights()
                )

            aggregated_model_weights = self.gen_aggreation(
                client_classification_model_weights_list, client_weights_list
            )

            for client_idx in range(self.args.client_num):
                self.client_list[client_idx].load_classification_model_weights(
                    aggregated_model_weights
                )
            if (round_idx + 1) % self.args.class_test_gap == 0:
                self.classification_model.load_state_dict(aggregated_model_weights[0])
                self.class_valid_test()

        pass

    def class_valid_test(self):
        self.classification_model = move2device(
            self.device, self.args.multi_gpu, self.classification_model
        )
        self.classification_model.eval()
        with torch.no_grad():
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 3.54]).to(self.device))
            if self.args.class_valid:
                loss = 0.0
                accurate_num = 0
                total_num = 0
                confusion_matrix = torch.zeros(
                    self.classification_dataset_valid.num_classes,
                    self.classification_dataset_valid.num_classes,
                )

                for batch in self.classification_dataloader_valid:
                    image, label = self.classification_dataset_valid.class_organize_batch(
                        batch, self.args, self.device
                    )
                    image = image.to(self.device)
                    label = label.to(self.device)
                    output = self.classification_model(image)
                    predictions = torch.argmax(output, dim=1)
                    accurate_num += (predictions == label).sum().item()
                    total_num += label.size(0)
                    loss += criterion(output, label).item()
                    for i in range(predictions.size(0)):
                        true_label = label[i].item()
                        predicted_label = predictions[i].item()
                        confusion_matrix[true_label][predicted_label] += 1
                precision = torch.diag(confusion_matrix) / confusion_matrix.sum(dim=0)
                recall = torch.diag(confusion_matrix) / confusion_matrix.sum(dim=1)
                f1 = 2 * precision * recall / (precision + recall)
                accuracy = (torch.diag(confusion_matrix).sum() / confusion_matrix.sum()).item()
                self.logger.log(f"Confusion matrix:{confusion_matrix.tolist()}")
                self.logger.log(f"Precision: {precision.tolist()}")
                self.logger.log(f"Recall: {recall.tolist()}")
                self.logger.log(f"F1: {f1.tolist()}")
                self.logger.log("Validation Average Accuracy: {:.4f}%".format(100 * accuracy))
                self.logger.log(
                    "Validation Average Loss: {:.4f}".format(
                        loss / len(self.classification_dataloader_valid)
                    )
                )

            loss = 0.0
            accurate_num = 0
            total_num = 0
            confusion_matrix = torch.zeros(
                self.classification_dataset_valid.num_classes,
                self.classification_dataset_valid.num_classes,
            )
            for batch in self.classification_dataloader_test:

                image, label = self.classification_dataset_test.class_organize_batch(
                    batch, self.args, self.device
                )
                image = image.to(self.device)
                label = label.to(self.device)
                output = self.classification_model(image)
                predictions = torch.argmax(output, dim=1)
                accurate_num += (predictions == label).sum().item()
                total_num += label.size(0)
                loss += criterion(output, label).item()
                for i in range(predictions.size(0)):
                    true_label = label[i].item()
                    predicted_label = predictions[i].item()
                    confusion_matrix[true_label][predicted_label] += 1
            precision = torch.diag(confusion_matrix) / confusion_matrix.sum(dim=0)
            recall = torch.diag(confusion_matrix) / confusion_matrix.sum(dim=1)
            f1 = 2 * precision * recall / (precision + recall)
            accuracy = (torch.diag(confusion_matrix).sum() / confusion_matrix.sum()).item()
            self.logger.log(
                "Validation Average Accuracy: {:.4f}%".format(100 * accurate_num / total_num)
            )
            self.logger.log(f"Confusion matrix:{confusion_matrix.tolist()}")
            self.logger.log(f"Precision: {precision.tolist()}")
            self.logger.log(f"Recall: {recall.tolist()}")
            self.logger.log(f"F1: {f1.tolist()}")
            self.logger.log("Test Average Accuracy: {:.4f}%".format(100 * accuracy))
            self.logger.log(
                "Test Average Loss: {:.4f}".format(loss / len(self.classification_dataloader_test))
            )
            self.classification_model = move2cpu(self.classification_model)

    def run(self):
        pass

    def debug(self):
        # self.args.multi_gpu = False
        if self.args.use_generator:
            self.gen_resume()
        self.classification_workflow()


if __name__ == "__main__":
    server = FedAvgServer()
    server.args.debug = True
    if server.args.debug:
        server.debug()
    else:
        server.generation_workflow()
    # server.debug()
