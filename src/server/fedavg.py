import pickle
import sys
import json
import os
import random
from pathlib import Path
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
import time
from typing import Dict, List, OrderedDict

import torch
from torchvision.transforms import ToTensor

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
        default=10,
        help="Number of global rounds for the generation task.",
    )
    parser.add_argument(
        "--gen-epochs",
        type=int,
        default=10,
        help="Number of local training epochs for the generation model.",
    )
    parser.add_argument(
        "--gen-lr", type=float, default=0.0002, help="Learning rate for the generation model."
    )
    parser.add_argument(
        "--gen-batch-size", type=int, default=2, help="Batch size for the generation model."
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
        # load args & set random seed
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

        self.device = get_best_device(self.args.use_cuda)
        self.client = FedAvgClient(self.args, self.device)
        # initialize test dataset
        self.transform = ToTensor()
        self.test_dataset = DATASETS[self.args.dataset](self.args, transform=self.transform)

    def generation_workflow(self):
        client_model_weights_list = []
        for round_idx in range(self.args.gen_round):
            for i in range(self.args.client_num):
                client_model_weights_list.append(self.client.train_gen_model(i))

        pass

    def classification(self):
        pass

    def run(self):
        self.generation()
        self.classification()


if __name__ == "__main__":
    server = FedAvgServer()
    server.generation_workflow()
