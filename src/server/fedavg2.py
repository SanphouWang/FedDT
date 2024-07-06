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
from multiprocessing import Manager
from ray.util.multiprocessing import Pool
import torch.nn as nn

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())

from data.utils.datasets import CALLATE_FNC, DATASETS

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
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["brats2019", "adni_roi"],
        type=str,
        default="adni_roi",
        help="dataset to use",
    )
    parser.add_argument("-s", "--seed-server", type=int, default=42, help="random seed")
    parser.add_argument("--use-cuda", type=int, default=1)
    parser.add_argument("--multi-gpu", type=bool, default=False, help="whether to use multi-gpu")
    parser.add_argument("--save_log", type=int, default=1)
    parser.add_argument("--debug", default=False, type=bool, help="whether to debug the code")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="classification/fortest/",
        help="output directory",
    )
    parser.add_argument(
        "--task", choices=["classification", "generation"], default="generation", type=str
    )
    parser.add_argument("--log-name", default="class", type=str)
    parser.add_argument(
        "--preprocessed-file-directory",
        default="/mnt/hardDisk1/wwmm/sanphou/FedDT/out/Centralized/adni_roi/2024-06-03-10:27:49/classification/k=3/ratio_both=0.8/sigma=0.2/without_generator/checkpoint/classifier1.pt",
        type=str,
    )
    """
    # parameters for generation
    """
    # parser.add_argument(
    #     "--embed-p2p", type=bool, default=False, help="Whether to embed pixel2pixel method"
    # )

    parser.add_argument(
        "--gen-valid", "-gv", default=False, type=bool, help="whether to use validation set"
    )
    parser.add_argument(
        "--gen-model",
        type=str,
        default="pixel2pixel",
        choices=["cyclegan", "pixel2pixel"],
        help="Model architecture for the generation task.",
    )
    parser.add_argument(
        "--gen-round",
        type=int,
        default=10,
        help="Number of global rounds for the generation task.",
    )
    parser.add_argument(
        "--gen-samelr-round",
        type=int,
        default=10,
        help="Number of rounds with the same learning rate",
    )
    parser.add_argument(
        "--gen-epochs",
        type=int,
        default=5,
        help="Number of local training epochs for the generation model.",
    )
    parser.add_argument(
        "--gen-lr", type=float, default=0.001, help="Learning rate for the generator."
    )
    parser.add_argument(
        "--dis-lr", type=float, default=0.001, help="Learning rate for the discriminator"
    )
    parser.add_argument(
        "--gen-batch-size", type=int, default=32, help="Batch size for the generation model."
    )
    parser.add_argument(
        "--gen-test-gap", "-gtg", type=int, default=1, help="The gap between two test rounds"
    )
    # parser.add_argument('--gen-opt',choices=['adam', 'sgd'],default='sgd')
    parser.add_argument(
        "--upload-dis", type=bool, default=False, help="whether to upload discriminator weights"
    )
    parser.add_argument(
        "--gen-eval-train",
        default=10,
        type=int,
        help="whether to evaluate the generation model on the training set",
    )
    parser.add_argument("--gen-beta1", type=float, default=0.5, help="parameter for Adam")
    parser.add_argument("--gen-beta2", type=float, default=0.999, help="parameter for Adam")
    parser.add_argument(
        "--lambda-gan", type=float, default=5, help="parameter for the GAN loss0 function"
    )
    parser.add_argument(
        "--lambda-identity",
        type=float,
        default=0.0,
        help="parameter for the identity loss function",
    )
    parser.add_argument(
        "--lambda-cycle", type=float, default=100.0, help="parameter for the cycle loss function"
    )
    parser.add_argument(
        "--lambda-paired", type=float, default=10, help="use paired loss for better generation"
    )
    parser.add_argument(
        "--lambda-pixel",
        type=float,
        default=100.0,
        help="parameter for the pixel loss function in pexel2pixel",
    )
    parser.add_argument(
        "--lambda-label", default=50, type=float, help="parameter for the label loss function"
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
    parser.add_argument(
        "--embed-label",
        default=True,
        type=bool,
        help="whether embed label discriminator into the training process of gernerator",
    )
    """
    # paramters for classification
    """

    parser.add_argument(
        "--class-model",
        type=str,
        choices=["dualmodalitymlp", "mlp_classifier"],
        default="mlp_classifier",
        help="Model architecture for the classification task.",
    )
    parser.add_argument(
        "--class-batch-size", type=int, default=128, help="Batch size for the classification model."
    )
    parser.add_argument(
        "--lr-same-epochs",
        default=100,
        type=int,
        help="Number of epochs with the same initial learning rate",
    )
    parser.add_argument(
        "--class-epochs",
        type=int,
        default=200,
        help="Number of local training epochs for the classification model.",
    )
    parser.add_argument(
        "--class-round",
        type=int,
        default=40,
        help="Number of local training epochs for the classification model.",
    )
    parser.add_argument(
        "--class-lr",
        type=float,
        default=0.01,
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
        "--class-test-gap", type=int, default=40, help="The gap between two test rounds"
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
        self.out_dir = (
            OUT_DIR
            / self.algo
            / self.args.dataset
            / self.args.out_dir
            / (self.args.log_name if self.args.log_name else begin_time)
        )
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
            self.classification_dataset_test,
            batch_size=self.args.class_batch_size,
            shuffle=False,
            collate_fn=lambda batch: CALLATE_FNC[self.args.dataset](
                batch, self.args, "classification"
            ),
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
                collate_fn=lambda batch: CALLATE_FNC[self.args.dataset](
                    batch, self.args, "classification"
                ),
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

        # calculate weights for each client for model aggregation
        client_datanum_list = [
            dataset.data_amount("generation") for dataset in self.generation_dataset_train
        ]
        total_datanum = sum(client_datanum_list)
        client_weights_list = [datanum / total_datanum for datanum in client_datanum_list]
        # unify generation model weights
        if self.args.gen_resume:
            self.gen_resume()
        else:
            for client_idx in range(self.args.client_num):
                self.client_list[client_idx].load_generation_model_weights(
                    self.client_list[0].get_generation_model_weights()
                )
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
                if self.args.dataset != "adni_roi":
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
                            client_weights_list[client_idx] * model_dict[key]
                        )
                    else:
                        aggregated_model_dict_list[model_idx][key] += (
                            client_weights_list[client_idx] * model_dict[key]
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
        self.generation_model.download_weights(aggregated_model_dict_list)

    def classification_workflow(self):
        client_datanum_list = [
            dataset.data_amount("classification") for dataset in self.classification_dataset_train
        ]
        total_datanum = sum(client_datanum_list)
        client_weights_list = [datanum / total_datanum for datanum in client_datanum_list]
        for client_idx in range(self.args.client_num):
            self.client_list[client_idx].load_classification_model_weights(
                [self.classification_model.state_dict()]
            )
        for round_idx in range(self.args.class_round):
            self.logger.log(
                "=" * 20, "Round:", round_idx, "Classification Start Training", "=" * 20
            )
            client_classification_model_weights_map = {}
            for client_idx in range(self.args.client_num):
                self.logger.log(local_time(), " Client", client_idx, "Training Start")
                self.client_list[client_idx].classification_workflow()
                client_classification_model_weights_map[client_idx] = self.client_list[
                    client_idx
                ].get_classification_model_weights()

            aggregated_model_weights = self.classification_aggregation(
                client_classification_model_weights_map, client_weights_list
            )

            for client_idx in range(self.args.client_num):
                self.client_list[client_idx].load_classification_model_weights(
                    aggregated_model_weights
                )
            if (round_idx + 1) % self.args.class_test_gap == 0:
                self.classification_model.load_state_dict(aggregated_model_weights[0])
                self.class_valid_test_workflow()

        pass

    def client_classification_process(self, client_idx, client_classification_model_weights_map):
        self.client_list[client_idx].classification_workflow()
        client_classification_model_weights_map[client_idx] = self.client_list[
            client_idx
        ].get_classification_model_weights()

    def class_valid_test_workflow(self):
        self.classification_model = move2device(
            self.device, self.args.multi_gpu, self.classification_model
        )
        self.classification_model.eval()
        with torch.no_grad():
            if self.args.class_valid:
                self.logger.log(local_time(), " Validation Start")
                self.classification_test(self.classification_dataloader_valid)
            self.logger.log(local_time(), " Test Start")
            self.classification_test(self.classification_dataloader_test)
        self.classification_model = move2cpu(self.classification_model)

    def classification_test(self, dataloader):
        criterion = nn.CrossEntropyLoss().to(self.device)
        """
        real modality0 + real modality1
        """
        label_all = []
        prediction_all = []
        for batch in dataloader:
            batch, _, _, _ = batch
            modality0_volumes = batch["modality0"].to(self.device)
            modality1_volumes = batch["modality1"].to(self.device)
            predict_result = self.classification_model(modality0_volumes, modality1_volumes)
            predict_result = torch.argmax(predict_result, dim=1)
            label = batch["label"]
            prediction_all.extend(predict_result.cpu().numpy())
            label_all.extend(label.cpu().numpy())
        cm, precision, recall, f1_score, acc = FedAvgClient.classification_metric(
            label_all, prediction_all
        )
        self.logger.log("*****real modality0 + real modality1*****")
        self.logger.log(f"Confusion matrix:\n{cm}")
        self.logger.log(f"Precision={precision*100:.2f}%")
        self.logger.log(f"Recall={recall*100:.2f}%")
        self.logger.log(f"F1={f1_score*100:.2f}%")
        self.logger.log(f"Accuracy={acc*100:.2f}%")
        if self.args.use_generator:
            self.generation_model.move2device()
            self.generation_model.gen_021.eval()
            self.generation_model.gen_120.eval()
            """
            real modality0 + generated modality1
            """
            label_all = []
            prediction_all = []
            for batch in dataloader:
                batch, _, _, _ = batch
                modality0_volumes = batch["modality0"].to(self.device)
                fake_modality1 = self.generation_model.gen_021(modality0_volumes)
                predict_result = self.classification_model(modality0_volumes, fake_modality1)
                predict_result = torch.argmax(predict_result, dim=1)
                label = batch["label"]
                prediction_all.extend(predict_result.cpu().numpy())
                label_all.extend(label.cpu().numpy())
            cm, precision, recall, f1_score, acc = FedAvgClient.classification_metric(
                label_all, prediction_all
            )
            self.logger.log("*****real modality0 + generated modality1*****")
            self.logger.log(f"Confusion matrix:\n{cm}")
            self.logger.log(f"Precision={precision*100:.2f}%")
            self.logger.log(f"Recall={recall*100:.2f}%")
            self.logger.log(f"F1={f1_score*100:.2f}%")
            self.logger.log(f"Accuracy={acc*100:.2f}%")
            """
            generated modality0 + real modality1
            """
            label_all = []
            prediction_all = []
            for batch in dataloader:
                batch, _, _, _ = batch
                modality1_volumes = batch["modality1"].to(self.device)
                fake_modality0 = self.generation_model.gen_120(modality1_volumes)
                predict_result = self.classification_model(fake_modality0, modality1_volumes)
                predict_result = torch.argmax(predict_result, dim=1)
                label = batch["label"]
                prediction_all.extend(predict_result.cpu().numpy())
                label_all.extend(label.cpu().numpy())
            cm, precision, recall, f1_score, acc = FedAvgClient.classification_metric(
                label_all, prediction_all
            )
            self.logger.log("*****generated modality0 + real modality1*****")
            self.logger.log(f"Confusion matrix:\n{cm}")
            self.logger.log(f"Precision={precision*100:.2f}%")
            self.logger.log(f"Recall={recall*100:.2f}%")
            self.logger.log(f"F1={f1_score*100:.2f}%")
            self.logger.log(f"Accuracy={acc*100:.2f}%")
            self.generation_model.move2cpu()

    def classification_aggregation(self, state_dict_map, weights_list: List) -> List:
        weighted_average_dict = {}
        for key in state_dict_map[0][0]:
            weighted_average_dict[key] = torch.zeros_like(state_dict_map[0][0][key])
        for client_idx in state_dict_map.keys():
            for key in state_dict_map[0][0]:
                weighted_average_dict[key] += (
                    state_dict_map[client_idx][0][key] * weights_list[client_idx]
                )
        return [weighted_average_dict]

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
