from collections import OrderedDict
from pathlib import Path
from typing import List
import torch

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
from data.utils.datasets import DATASETS

import torch.optim as optim

from src.utils.tools import move2device, move2cpu, get_best_device
from src.model.model_tools import get_model_arch
from tqdm import tqdm
import torch.nn as nn


class FedAvgClient:
    def __init__(self, args, dataset, logger):
        self.args = args
        self.dataset = dataset
        self.logger = logger
        self.client_num = self.args.client_num
        self.dataloader4generation = DataLoader(
            self.dataset, batch_size=self.args.gen_batch_size, shuffle=True
        )
        self.generation_model = get_model_arch(self.args.gen_model)(args, logger)
        self.dataloader4classification = None
        self.classification_model = None
        self.device = None

    def choose_device(self):
        if self.device is None:
            self.device = get_best_device(self.args.use_cuda)

    def generation_workflow(self, round_idx):
        self.choose_device()
        self.generation_model.device = self.device
        self.generation_model.move2device()
        for _ in range(self.args.gen_epochs):
            self.generation_model.train_epoch(self.dataloader4generation)
            self.generation_model.update_learning_rate()
        self.generation_model.move2cpu()

    def get_generation_model_weights(self) -> List[List[OrderedDict]]:
        # return weights of models on each client
        # since each model may have different submodels (e.g. CycleGAN has two generator and discriminator)
        # each element in the returned list is a list of weights of submodels (e.g. [weights of generator1, weights of generator2, ...])
        return self.generation_model.get_weights()

    def load_generation_model_weights(self, aggregated_model_weights, resume=False):
        self.generation_model.download_weights(aggregated_model_weights, resume)

    def classification_workflow(
        self,
    ):
        self.choose_device()
        # Train the classification model locally
        if self.dataloader4classification is None:
            self.dataloader4classification = DataLoader(
                self.dataset, batch_size=self.args.class_batch_size, shuffle=True
            )
        if self.classification_model is None:
            self.classification_model = get_model_arch(self.args.class_model)()
            self.classification_learning_rate = self.args.class_lr
        self.classification_optimizer = optim.Adam(
            self.classification_model.parameters(),
            lr=self.classification_learning_rate,
            # momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 3.54]).to(self.device))
        self.classification_model = move2device(
            self.device, self.args.multi_gpu, self.classification_model
        )
        if self.args.use_generator:
            generator_021, generator_120 = self.generation_model.get_generator()
            generator_021.eval()
            generator_120.eval()
            generator_021 = move2device(self.device, self.args.multi_gpu, generator_021)
            generator_120 = move2device(self.device, self.args.multi_gpu, generator_120)
        self.classification_model.train()
        for epoch in range(self.args.class_epochs):
            for batch in self.dataloader4classification:
                if self.args.use_generator:
                    image, label = self.dataloader4classification.dataset.class_organize_batch(
                        batch, self.args, self.device, generator_021, generator_120
                    )
                else:
                    image, label = self.dataloader4classification.dataset.class_organize_batch(
                        batch, self.args, self.device
                    )
                image = image.to(self.device)
                label = label.to(self.device)
                # forward pass
                output = self.classification_model(image)
                # calculate loss
                loss = criterion(output, label)
                # backward pass
                self.classification_optimizer.zero_grad()
                loss.backward()
                self.classification_optimizer.step()
            # self.classification_scheduler.step()
            # Evaluate accuracy and loss on the training set
        self.evaluate_classification()
        self.update_classification_learning_rate()
        self.classification_model = move2cpu(self.classification_model)
        if self.args.use_generator:
            move2cpu(generator_021)
            move2cpu(generator_120)
        pass

    def evaluate_classification(self):
        # Test the classification model locally
        with torch.no_grad():
            self.classification_model.eval()
            criterion = nn.CrossEntropyLoss()
            confusion_matrix = torch.zeros(self.dataset.num_classes, self.dataset.num_classes)
            for batch in self.dataloader4classification:
                image, label = self.dataloader4classification.dataset.class_organize_batch(
                    batch, self.args, self.device
                )
                image = image.to(self.device)
                label = label.to(self.device)
                output = self.classification_model(image)
                loss = criterion(output, label)
                predictions = torch.argmax(output, dim=1)
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
            self.logger.log(f"Evaluation: Accuracy={accuracy*100:.4f}%, Loss={loss.item():.4f}")
        pass

    def get_classification_model_weights(self) -> List[OrderedDict]:
        return [self.classification_model.state_dict()]

    def load_classification_model_weights(self, aggregated_model_weights):
        self.classification_model.load_state_dict(aggregated_model_weights[0])

    def update_classification_learning_rate(self):
        step = self.args.class_lr / self.args.class_round
        self.classification_learning_rate -= step
        pass
