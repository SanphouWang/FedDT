from collections import OrderedDict
from pathlib import Path
from typing import List
import torch

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
from data.utils.datasets import DATASETS

import torch.optim as optim

from src.utils.tools import move2device, move2cpu
from src.model.model_tools import get_model_arch
from tqdm import tqdm
import torch.nn as nn


class FedAvgClient:
    def __init__(self, args, device, logger):
        self.args = args
        self.device = device
        self.logger = logger
        self.client_num = self.args.client_num
        self.transform = ToTensor()
        self.dataset_class = DATASETS[self.args.dataset]
        self.dataset_list = [
            DATASETS[self.args.dataset](self.args, client_id=i) for i in range(self.client_num)
        ]
        # initialize model for generation
        self.train_loader_list = [
            DataLoader(self.dataset_list[i], batch_size=self.args.gen_batch_size, shuffle=True)
            for i in range(self.client_num)
        ]
        self.class_loader_list = [
            DataLoader(self.dataset_list[i], batch_size=self.args.class_batchsize, shuffle=True)
            for i in range(self.client_num)
        ]
        # gen model refer to generation model class, e.g. CycleGAN class
        self.gen_model_list = [
            get_model_arch(self.args.gen_model)(self.args, self.device, train_loader, self.logger)
            for train_loader in self.train_loader_list
        ]
        # list classification models
        self.class_model_list = [
            get_model_arch(self.args.class_model)(self.args)
            for train_loader in self.train_loader_list
        ]

    def get_client_weights(self) -> List[float]:
        # generate weight of each client for model aggregation in server
        dataset_len_list = [dataset.get_len() for dataset in self.dataset_list]
        total_num = sum(dataset_len_list)
        weights_list = [len_dataset / total_num for len_dataset in dataset_len_list]
        return weights_list

    def gen_get_model_weights(self) -> List[List[OrderedDict]]:
        # return weights of models on each client
        # since each model may have different submodels (e.g. CycleGAN has two generator and discriminator)
        # each element in the returned list is a list of weights of submodels (e.g. [weights of generator1, weights of generator2, ...])
        return [model.get_weights() for model in self.gen_model_list]

    def gen_download_model(self, aggregated_model_dict_list, resume=False):
        for model in self.gen_model_list:
            model.download_weights(aggregated_model_dict_list, resume)
        pass

    def gen_train_model(self, client_idx):
        # Train the generation model locally
        self.gen_model_list[client_idx].move2device()
        self.logger.log(f"client {client_idx} begin. ")

        for i in range(self.args.gen_epochs):
            self.gen_model_list[client_idx].train_epoch()
            self.gen_model_list[client_idx].update_learning_rate()
        # if self.args.gen_eval_train:
        #     result_after = self.gen_model_list[client_idx].evaluate_training()
        # self.logger.log(f"client {client_idx} finish. ")
        # if self.args.gen_eval_train:
        #     self.logger.log(
        #         "After: "
        #         + " ".join([f"{key}: {float(value):.5f}. " for key, value in result_after.items()]),
        #     )
        self.gen_model_list[client_idx].move2cpu()

    def class_train_model(self, client_idx):
        # Train the classification model locally
        criterion = nn.CrossEntropyLoss()
        dataloader = self.class_loader_list[client_idx]
        class_model = self.class_model_list[client_idx]
        optimizer = optim.SGD(
            class_model.parameters(),
            lr=self.args.class_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        class_model = move2device(self.device, self.args.multi_gpu, class_model)
        if self.args.use_generator:
            generator_021, generator_120 = self.gen_model_list[client_idx].get_generator()
            generator_021.eval()
            generator_120.eval()
            generator_021 = move2device(self.device, self.args.multi_gpu, generator_021)
            generator_120 = move2device(self.device, self.args.multi_gpu, generator_120)
        class_model.train()

        for epoch in range(self.args.class_epochs):

            for batch in dataloader:
                if self.args.use_generator:
                    image, label = self.dataset_class.class_organize_batch(
                        batch, self.args, self.device, generator_021, generator_120
                    )
                else:
                    image, label = self.dataset_class.class_organize_batch(
                        batch, self.args, self.device
                    )
                # forward pass
                output = class_model(image)
                # calculate loss
                loss = criterion(output, label)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Evaluate accuracy and loss on the training set
            # class_model.eval()
            total_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in dataloader:
                    image, label = self.dataset_class.class_organize_batch(
                        batch, self.args, self.device
                    )
                    output = class_model(image)
                    loss = criterion(output, label)
                    total_loss += loss.item()
                    predictions = torch.argmax(output, dim=1)
                    total += label.size(0)
                    correct += (predictions == label).sum().item()
            accuracy = 100.0 * correct / total
            average_loss = total_loss / len(dataloader)
            self.logger.log(
                f"Clinet {client_idx} Training Set: Accuracy={accuracy:.4f}%, Loss={average_loss:.4f}"
            )
        move2cpu(class_model)
        if self.args.use_generator:
            move2cpu(generator_021)
            move2cpu(generator_120)
        pass

    def class_test(self):
        # Test the classification model locally
        pass

    def class_get_model_weights(self) -> List[List[OrderedDict]]:
        return [[model.state_dict()] for model in self.class_model_list]

    def class_download_model(self, aggregated_model_dict_list, resume=False):
        for model in self.class_model_list:
            model.load_state_dict(aggregated_model_dict_list[0])
