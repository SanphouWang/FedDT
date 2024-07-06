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
import pandas as pd

import torch
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import numpy as np
from rich.console import Console
from rich.progress import track
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import torchvision.transforms as transforms
import wandb

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
from src.server.fedavg2 import local_time
from data.utils.datasets import ADNI_ROI, CALLATE_FNC, DATASETS

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
from src.server.fedavg2 import get_fedavg_argparser, local_time


class Centralized_ANDI_ROI(Dataset):
    def __init__(
        self,
        args,
        transform=None,
        mode="train",
    ) -> None:
        super().__init__()
        self.args = args
        self.transform = transform if transform else transforms.Compose([])
        self.mode = mode
        with open(
            os.path.join(args.data_path, args.preprocessed_file_directory, "patient_partition.pkl"),
            "rb",
        ) as f:
            self.patient_partition = pickle.load(f)
        self.data = pd.read_csv(os.path.join(args.data_path, "preprocessed.csv"))
        self.Statistic3M = pd.read_csv(os.path.join(args.data_path, "Statistic3M.csv"))
        self.label_mapping = {"MCI": 0, "CN": 1, "AD": 2, "Empty": 3}
        # judge the type of dataset
        self.num_classes = 3
        self.training_set = {
            self.args.modality0: [],
            self.args.modality1: [],
            "complete_patient": [],
            "patient": [],
        }
        for client_idx in range(1):
            self.training_set[self.args.modality0].extend(
                self.patient_partition[client_idx][self.args.modality0]
            )
            self.training_set[self.args.modality1].extend(
                self.patient_partition[client_idx][self.args.modality1]
            )
            self.training_set["complete_patient"].extend(
                self.patient_partition[client_idx]["complete_patient"]
            )
        self.training_set["patient"] = list(
            (
                set(self.training_set[self.args.modality0])
                | set(self.training_set[self.args.modality1])
            )
        )
        self.test_set = self.patient_partition["test"]
        self.valid_set = self.patient_partition["valid"]

    def __getitem__(self, index):
        if self.mode == "train":
            patient_id = self.training_set["patient"][index]
        elif self.mode == "test":
            patient_id = self.test_set["patient"][index]
        elif self.mode == "valid":
            patient_id = self.valid_set["patient"][index]
        else:
            raise ValueError("Invalid mode")
        patient_data = self.data[self.data["IID"] == patient_id]
        label = self.Statistic3M[self.Statistic3M["IID"] == patient_id]["diagnosis"].values[0]
        label = self.label_mapping[label]
        label = torch.tensor(label)
        if self.mode == "train":
            modality0 = np.zeros(90)
            modality1 = np.zeros(90)
            complete_flag = 0
            if patient_id in self.training_set[self.args.modality0]:
                modality0 = patient_data[patient_data["modality"] == self.args.modality0].values[
                    0, 2:
                ]
                complete_flag = 1
                modality = self.args.modality0
            if patient_id in self.training_set[self.args.modality1]:
                modality1 = patient_data[patient_data["modality"] == self.args.modality1].values[
                    0, 2:
                ]
                complete_flag = 1 * complete_flag
                if complete_flag == 1:
                    modality = "both"
                else:
                    modality = self.args.modality1
            modality0 = torch.tensor(modality0.astype(np.float32))
            modality1 = torch.tensor(modality1.astype(np.float32))
        else:
            modality = "both"
            modality0 = patient_data[patient_data["modality"] == self.args.modality0].values[0, 2:]
            modality1 = patient_data[patient_data["modality"] == self.args.modality1].values[0, 2:]
            modality0 = torch.tensor(modality0.astype(np.float32))
            modality1 = torch.tensor(modality1.astype(np.float32))

        return {
            "modality0": modality0,
            "modality1": modality1,
            "modality": modality,
            "label": label,
            "IID": patient_id,
        }

    @staticmethod
    def post_organize(volumes):
        return volumes

    def __len__(self):
        if self.mode == "train":
            return len(self.training_set["patient"])
        elif self.mode == "test":
            return len(self.test_set["patient"])
        elif self.mode == "valid":
            return len(self.valid_set["patient"])
        else:
            raise ValueError("Invalid mode")


def collate_fn(batch, args, task="classification"):
    assert task in ["generation", "classification"]
    batch = default_collate(batch)
    if task == "generation":
        modality0_indices, modality1_indices, paired_in_modality0, paired_in_modality1 = (
            ADNI_ROI.generate_indices(args, batch)
        )
        return batch, modality0_indices, modality1_indices, paired_in_modality0, paired_in_modality1
    elif task == "classification":
        indices = torch.tensor([i for i, item in enumerate(batch["label"]) if item != 3])
        for key in batch.keys():
            if isinstance(batch[key], list):
                batch[key] = [batch[key][i] for i in indices]
            else:
                batch[key] = batch[key][indices]
        modality0_indices, modality1_indices, _, _ = ADNI_ROI.generate_indices(args, batch)
        complete_indices = list(set(modality0_indices) & set(modality1_indices))
        single_modality0_indices = [i for i in modality0_indices if i not in complete_indices]
        single_modality1_indices = [i for i in modality1_indices if i not in complete_indices]
        return batch, single_modality0_indices, single_modality1_indices, complete_indices


class Centralized:
    def __init__(self, algo="Centralized", args: Namespace = None):
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
            PROJECT_DIR
            / "data"
            / self.args.dataset
            / self.args.preprocessed_file_directory
            / "args.pkl",
            "rb",
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
        Initialize Dataset
        """
        # Dataset

        self.classification_training_set = Centralized_ANDI_ROI(self.args, mode="train")
        self.classification_test_set = Centralized_ANDI_ROI(self.args, mode="test")
        self.classification_valid_set = Centralized_ANDI_ROI(self.args, mode="valid")
        self.generation_training_dataloader = DataLoader(
            self.classification_training_set,
            batch_size=self.args.gen_batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.args, task="generation"),
        )
        self.classification_training_dataloader = DataLoader(
            self.classification_training_set,
            batch_size=self.args.class_batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.args, task="classification"),
        )
        self.classification_test_dataloader = DataLoader(
            self.classification_test_set,
            batch_size=self.args.class_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, self.args, task="classification"),
        )
        self.generation_test_dataloader = DataLoader(
            self.classification_test_set,
            batch_size=self.args.class_batch_size,
            shuffle=False,
        )
        self.classification_valid_dataloader = DataLoader(
            self.classification_valid_set,
            batch_size=self.args.class_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, self.args, task="classification"),
        )
        self.generation_valid_dataloader = DataLoader(
            self.classification_valid_set,
            batch_size=self.args.class_batch_size,
            shuffle=False,
        )
        self.device = get_best_device(self.args.use_cuda)
        self.classification_model_both = get_model_arch(
            "dualmodalitymlp"
        )()  # classifier for both modalities
        self.classification_model_m0 = get_model_arch(
            "singlemodalitymlp"
        )()  # classifier for modality 0
        self.classification_model_m1 = get_model_arch(
            "singlemodalitymlp"
        )()  # classifier for modality 1
        self.generation_model = get_model_arch(self.args.gen_model)(self.args, self.logger)

    def generation_workflow(self):
        self.generation_model.device = self.device
        self.generation_model.move2device()

        for round in range(self.args.gen_round):
            self.logger.log(local_time(), f"Round {round+1}/{self.args.gen_round}")
            self.generation_model.train_epoch(self.generation_training_dataloader)
            self.generation_model.update_learning_rate()
            self.generation_model.current_round += 1
            if self.args.gen_save_checkpoint:
                self.gen_save_checkpoint(round, self.generation_model.get_weights())
        self.gen_valid_test()
        self.generation_model.move2cpu()

    def gen_valid_test(self):
        self.logger.log("Validation set:")
        valid_result = self.generation_model.test(self.generation_valid_dataloader)
        for key, value in valid_result.items():
            self.logger.log(f"{key}: {value:.4f}")
        self.logger.log("Test set:")
        test_result = self.generation_model.test(self.generation_test_dataloader)
        for key, value in test_result.items():
            self.logger.log(f"{key}: {value:.4f}")

    def gen_save_checkpoint(self, round, weights):
        path = os.path.join(self.out_dir, "checkpoint")
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(
            {"round": round, "model_state_dict_list": weights},
            os.path.join(path, "checkpoint.pt"),
        )

    def gen_resume(self):
        checkpoint = torch.load(self.args.gen_resume)
        if not self.args.only_resume_model:
            self.gen_round_begin = checkpoint["round"] + 1
        aggregated_model_dict_list = checkpoint["model_state_dict_list"]
        self.generation_model.download_weights(aggregated_model_dict_list, resume=True)

    def classification_workflow(self):
        self.classification_initialization()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 2.0])).to(self.device)
        for epoch in range(self.args.class_epochs):
            self.logger.log(local_time(), f"Epoch {epoch+1}/{self.args.class_epochs}")
            self.classification_model_both.train()
            self.classification_model_m0.train()
            self.classification_model_m1.train()

            for batch in self.classification_training_dataloader:
                predict_result, label = self.classification_forward(batch)
                loss_classification = criterion(predict_result, label)
                for opt in self.classification_optimizer_list:
                    opt.zero_grad()
                loss_classification.backward()
                for opt in self.classification_optimizer_list:
                    opt.step()

            for scheduler in self.scheduler_list:
                scheduler.step()
            # Evaluate accuracy and loss on the training set
            if (epoch + 1) % self.args.class_test_gap == 0:
                wandb.log({"epoch": epoch})
                self.logger.log("=" * 10, "Validation set:", "=" * 10)
                validation_result = self.evaluate_classification(
                    self.classification_valid_dataloader
                )
                log_data = {}
                for test_set in validation_result.index:
                    for metric in validation_result.columns:
                        log_data[f"valid/{test_set}/{metric}"] = validation_result.loc[
                            test_set, metric
                        ]
                wandb.log(log_data)
                self.logger.log("=" * 10, "Test set:", "=" * 10)
                test_result = self.evaluate_classification(self.classification_test_dataloader)
                log_data = {}
                for test_set in test_result.index:
                    for metric in test_result.columns:
                        log_data[f"test/{test_set}/{metric}"] = test_result.loc[test_set, metric]
                wandb.log(log_data)
            # save model weights
        if not os.path.exists(os.path.join(self.out_dir, "checkpoint")):
            os.makedirs(os.path.join(self.out_dir, "checkpoint"))
        torch.save(
            self.classification_model_both.state_dict(),
            os.path.join(self.out_dir, "checkpoint", "classifier1.pt"),
        )
        torch.save(
            self.classification_model_m0.state_dict(),
            os.path.join(self.out_dir, "checkpoint", "classifier2.pt"),
        )
        torch.save(
            self.classification_model_m1.state_dict(),
            os.path.join(self.out_dir, "checkpoint", "classifier3.pt"),
        )

    def classification_forward(self, batch):
        batch, single_modality0_indices, single_modality1_indices, complete_indices = batch
        modality0_volumes = batch["modality0"]
        modality1_volumes = batch["modality1"]
        predict_result = []
        label = []
        if self.args.use_generator:
            modality0_volumes = modality0_volumes.to(self.device)
            modality1_volumes = modality1_volumes.to(self.device)
            if len(single_modality1_indices) > 0:
                modality1_real = modality1_volumes[single_modality1_indices]
                modality0_volumes[single_modality1_indices] = self.generator_120(modality1_real)
            if len(single_modality0_indices) > 0:
                modality0_real = modality0_volumes[single_modality0_indices]
                modality1_volumes[single_modality0_indices] = self.generator_021(modality0_real)
            predict_result.append(self.classification_model_m0(modality0_volumes))
            label.append(batch["label"])
            predict_result.append(self.classification_model_m1(modality1_volumes))
            label.append(batch["label"])
            predict_result.append(
                self.classification_model_both(modality0_volumes, modality1_volumes)
            )
            label.append(batch["label"])
        else:
            modality0_indices = single_modality0_indices + complete_indices
            modality1_indices = single_modality1_indices + complete_indices
            if len(modality0_indices) > 0:
                modality0_real = modality0_volumes[modality0_indices].to(self.device)
                predict_result.append(self.classification_model_m0(modality0_real))
                label.append(batch["label"][modality0_indices].to(self.device))
            if len(modality1_indices) > 0:
                modality1_real = modality1_volumes[modality1_indices].to(self.device)
                predict_result.append(self.classification_model_m1(modality1_real))
                label.append(batch["label"][modality1_indices].to(self.device))
            if len(complete_indices) > 0:
                modality0_real = modality0_volumes[complete_indices].to(self.device)
                modality1_real = modality1_volumes[complete_indices].to(self.device)
                predict_result.append(
                    self.classification_model_both(modality0_real, modality1_real)
                )
                label.append(batch["label"][complete_indices].to(self.device))
            # concatenated_volumes = torch.cat((modality0_volumes, modality1_volumes), dim=1)
        predict_result = torch.cat(predict_result, dim=0)
        label = torch.cat(label, dim=0).to(self.device)
        label = label.long()
        return predict_result, label

    def evaluate_classification(self, dataloader):
        # Test the classification model locally
        with torch.no_grad():
            self.classification_model_both.eval()
            self.classification_model_m0.eval()
            self.classification_model_m1.eval()
            criterion = nn.CrossEntropyLoss()
            """
            real modality0 + real modality1
            """
            label_all = []
            prediction_both = []
            prediction_m0 = []
            prediction_m1 = []
            csv_columns = {"Precision": [], "Recall": [], "F1": [], "Accuracy": []}
            for batch in dataloader:
                batch, single_modality0_indices, single_modality1_indices, complete_indices = batch
                modality0_volumes = batch["modality0"].to(self.device)
                modality1_volumes = batch["modality1"].to(self.device)
                pred_both = self.classification_model_both(modality0_volumes, modality1_volumes)
                pred_m0 = self.classification_model_m0(modality0_volumes)
                pred_m1 = self.classification_model_m1(modality1_volumes)
                prediction_both.extend(torch.argmax(pred_both, dim=1).cpu().numpy())
                prediction_m0.extend(torch.argmax(pred_m0, dim=1).cpu().numpy())
                prediction_m1.extend(torch.argmax(pred_m1, dim=1).cpu().numpy())
                label_all.extend(batch["label"].cpu().numpy())

            for index, prediction in enumerate([prediction_both, prediction_m0, prediction_m1]):

                cm, precision, recall, f1_score, acc = self.classification_metric(
                    label_all, prediction
                )
                if index == 0:
                    test_str = "*****Test with Both Modalities*****"
                elif index == 1:
                    test_str = "*****Test with Modality 0*****"
                elif index == 2:
                    test_str = "*****Test with Modality 1*****"
                self.logger.log(test_str)
                self.logger.log(f"Confusion matrix:\n{cm}")
                self.logger.log(f"Precision={precision*100:.2f}%")
                self.logger.log(f"Recall={recall*100:.2f}%")
                self.logger.log(f"F1={f1_score*100:.2f}%")
                self.logger.log(f"Accuracy={acc*100:.2f}%")
                csv_columns["Precision"].append(precision * 100)
                csv_columns["Recall"].append(recall * 100)
                csv_columns["F1"].append(f1_score * 100)
                csv_columns["Accuracy"].append(acc * 100)
            index = ["test set 1", "test set 2", "test set 3"]
            df = pd.DataFrame(csv_columns, index=index)
            df = df.round(2)
            df.to_csv(os.path.join(self.out_dir, "metrics.csv"))
        return df

    def classification_initialization(
        self,
    ):
        wandb.init(
            project="FedDT",
            name=self.args.log_name,
            group=self.args.out_dir,
            config=self.args,
            reinit=True,
            mode="offline",
        )
        self.classification_model_both = move2device(
            self.device, self.args.multi_gpu, self.classification_model_both
        )
        self.classification_model_m0 = move2device(
            self.device, self.args.multi_gpu, self.classification_model_m0
        )
        self.classification_model_m1 = move2device(
            self.device, self.args.multi_gpu, self.classification_model_m1
        )
        self.classification_optimizer_both = optim.SGD(
            self.classification_model_both.parameters(),
            lr=self.args.class_lr,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
            # betas=(self.args.gen_beta1, self.args.gen_beta2),
        )
        self.classification_optimizer_m0 = optim.SGD(
            self.classification_model_m0.parameters(),
            lr=self.args.class_lr,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
            # betas=(self.args.gen_beta1, self.args.gen_beta2),
        )
        self.classification_optimizer_m1 = optim.SGD(
            self.classification_model_m1.parameters(),
            lr=self.args.class_lr,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
            # betas=(self.args.gen_beta1, self.args.gen_beta2),
        )
        self.classification_optimizer_list = [
            self.classification_optimizer_both,
            self.classification_optimizer_m0,
            self.classification_optimizer_m1,
        ]

        def lambda_rule(epoch):

            return 1.0 - max((epoch + 1) - self.args.lr_same_epochs, 0) / (
                float(self.args.class_epochs) - self.args.lr_same_epochs
            )

        self.scheduler_both = optim.lr_scheduler.LambdaLR(
            self.classification_optimizer_both, lr_lambda=lambda_rule
        )
        self.scheduler_m0 = optim.lr_scheduler.LambdaLR(
            self.classification_optimizer_m0, lr_lambda=lambda_rule
        )
        self.scheduler_m1 = optim.lr_scheduler.LambdaLR(
            self.classification_optimizer_m1, lr_lambda=lambda_rule
        )
        self.scheduler_list = [self.scheduler_both, self.scheduler_m0, self.scheduler_m1]

        if self.args.use_generator:
            if self.args.gen_resume == "":
                raise ValueError(
                    "Please provide the path of the checkpoint if you want to use generator in classification task"
                )
            else:
                self.gen_resume()
                self.generator_021, self.generator_120 = self.generation_model.get_generator()
                self.generator_021 = move2device(
                    self.device, self.args.multi_gpu, self.generator_021
                )
                self.generator_120 = move2device(
                    self.device, self.args.multi_gpu, self.generator_120
                )
                self.generator_021.eval()
                self.generator_120.eval()

    @staticmethod
    def classification_metric(label_all, predicition_all):
        cm = confusion_matrix(label_all, predicition_all)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            label_all, predicition_all, average="macro"
        )
        acc = accuracy_score(label_all, predicition_all)
        return cm, precision, recall, f1_score, acc


if __name__ == "__main__":
    centralized = Centralized()
    if centralized.args.task == "generation":
        centralized.generation_workflow()
    else:
        centralized.classification_workflow()
