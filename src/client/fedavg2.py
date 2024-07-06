from collections import OrderedDict
from pathlib import Path
from typing import List
import torch

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
from data.utils.datasets import DATASETS, CALLATE_FNC

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
            self.dataset,
            batch_size=self.args.gen_batch_size,
            shuffle=True,
            collate_fn=lambda batch: CALLATE_FNC[self.args.dataset](batch, self.args),
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
        self.generation_model.current_round += 1
        self.generation_model.move2cpu()

    def get_generation_model_weights(self) -> List[OrderedDict]:
        # return weights of models on each client
        # since each model may have different submodels (e.g. CycleGAN has two generator and discriminator)
        # each element in the returned list is a list of weights of submodels (e.g. [weights of generator1, weights of generator2, ...])
        return self.generation_model.get_weights()

    def load_generation_model_weights(
        self, aggregated_model_weights: List[OrderedDict], resume=False
    ):
        self.generation_model.download_weights(aggregated_model_weights, resume)

    def classification_workflow(
        self,
    ):
        self.choose_device()
        # Train the classification model locally
        if self.dataloader4classification is None:
            self.dataloader4classification = DataLoader(
                self.dataset,
                batch_size=self.args.class_batch_size,
                shuffle=True,
                collate_fn=lambda batch: CALLATE_FNC[self.args.dataset](
                    batch, self.args, "classification"
                ),
            )
            if self.args.use_generator:
                self.generator_021, self.generator_120 = self.generation_model.get_generator()
                self.generator_021.eval()
                self.generator_120.eval()
        self.classification_optimizer = optim.SGD(
            self.classification_model.parameters(),
            lr=self.classification_learning_rate,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
            # betas=(self.args.gen_beta1, self.args.gen_beta2),
        )
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 2.0])).to(self.device)
        if self.args.use_generator:
            self.generator_021 = move2device(self.device, self.args.multi_gpu, self.generator_021)
            self.generator_120 = move2device(self.device, self.args.multi_gpu, self.generator_120)
        self.classification_model = move2device(
            self.device, self.args.multi_gpu, self.classification_model
        )

        for epoch in range(self.args.class_epochs):
            self.classification_model.train()
            for batch in self.dataloader4classification:
                predict_result, label = self.classification_forward(batch)
                if predict_result is not None and label is not None:
                    loss_classification = criterion(predict_result, label)
                    self.classification_optimizer.zero_grad()
                    loss_classification.backward()
                    self.classification_optimizer.step()
            # Evaluate accuracy and loss on the training set
        self.evaluate_classification()
        self.update_classification_learning_rate()
        self.classification_model = move2cpu(self.classification_model)
        if self.args.use_generator:
            move2cpu(self.generator_021)
            move2cpu(self.generator_120)
        pass

    def classification_forward(self, batch):
        batch, single_modality0_indices, single_modality1_indices, complete_indices = batch
        modality0_volumes = batch["modality0"]
        modality1_volumes = batch["modality1"]
        predict_result = []
        label = []
        if self.args.use_generator:
            if len(single_modality1_indices) > 0:
                modality1_real = modality1_volumes[single_modality1_indices].to(self.device)
                modality0_fake = self.generator_120(modality1_real)
                # concatenated_volumes = torch.cat((modality0_fake, modality1_real), dim=1)
                predict_result.append(self.classification_model(modality0_fake, modality1_real))
                label.append(batch["label"][single_modality1_indices])
                # loss_classification += criterion(predict_result, label_single)
            if len(single_modality0_indices) > 0:
                modality0_real = modality0_volumes[single_modality0_indices].to(self.device)
                modality1_fake = self.generator_021(modality0_real)
                # concatenated_volumes = torch.cat((modality0_real, modality1_fake), dim=1)
                predict_result.append(self.classification_model(modality0_real, modality1_fake))
                label.append(batch["label"][single_modality0_indices])
                # loss_classification += criterion(predict_result, label_single)
            if len(complete_indices) > 0:
                modality0_real = modality0_volumes[complete_indices].to(self.device)
                modality1_real = modality1_volumes[complete_indices].to(self.device)
                # concatenated_volumes = torch.cat((modality0_real, modality1_real), dim=1)
                predict_result.append(self.classification_model(modality0_real, modality1_real))
                label.append(batch["label"][complete_indices])
                # loss_classification += criterion(predict_result, label)
            predict_result = torch.cat(predict_result, dim=0)
            label = torch.cat(label, dim=0).to(self.device)
            label = label.long()
        else:
            if len(single_modality0_indices) > 0:
                modality0_real = modality0_volumes[single_modality0_indices].to(self.device)
                predict_result.append(self.classification_model(mod0=modality0_real))
                label.append(batch["label"][single_modality0_indices].to(self.device))
            if len(single_modality1_indices) > 0:
                modality1_real = modality1_volumes[single_modality1_indices].to(self.device)
                predict_result.append(self.classification_model(mod1=modality1_real))
                label.append(batch["label"][single_modality1_indices].to(self.device))
            if len(complete_indices) > 0:
                modality0_real = modality0_volumes[complete_indices].to(self.device)
                modality1_real = modality1_volumes[complete_indices].to(self.device)
                predict_result.append(self.classification_model(modality0_real, modality1_real))
                label.append(batch["label"][complete_indices].to(self.device))
            # concatenated_volumes = torch.cat((modality0_volumes, modality1_volumes), dim=1)
            try:
                predict_result = torch.cat(predict_result, dim=0)
                label = torch.cat(label, dim=0).to(self.device)
                label = label.long()
            except:
                return None, None
            # loss_classification = criterion(output, label)

        return predict_result, label

    def evaluate_classification(self):
        # Test the classification model locally
        with torch.no_grad():
            self.classification_model.eval()
            criterion = nn.CrossEntropyLoss()
            # confusion_matrix = torch.zeros(self.dataset.num_classes, self.dataset.num_classes)
            label_all = []
            predicition_all = []
            for batch in self.dataloader4classification:
                predict_result, label = self.classification_forward(batch)
                try:
                    loss = criterion(predict_result, label)
                    predictions = torch.argmax(predict_result, dim=1)
                    label_all.extend(label.cpu().numpy())
                    predicition_all.extend(predictions.cpu().numpy())
                except:
                    pass
            cm, precision, recall, f1_score, acc = self.classification_metric(
                label_all, predicition_all
            )
            self.logger.log(f"Confusion matrix:\n{cm}")
            self.logger.log(f"Precision: {precision*100:.2f}%")
            self.logger.log(f"Recall: {recall*100:.2f}%")
            self.logger.log(f"F1: {f1_score*100:.2f}%")
            self.logger.log(f"Accuracy={acc*100:.2f}%\nLoss={loss.item():.4f}")
        pass

    @staticmethod
    def classification_metric(label_all, predicition_all):
        cm = confusion_matrix(label_all, predicition_all)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            label_all, predicition_all, average="macro"
        )
        acc = accuracy_score(label_all, predicition_all)
        return cm, precision, recall, f1_score, acc

    def get_classification_model_weights(self) -> List[OrderedDict]:
        return [self.classification_model.state_dict()]
        # return [self.classification_model.parameters()]

    def load_classification_model_weights(self, aggregated_model_weights: List[OrderedDict]):
        if self.classification_model is None:
            self.classification_model = get_model_arch(self.args.class_model)()
            self.classification_learning_rate = self.args.class_lr
        self.classification_model.load_state_dict(aggregated_model_weights[0])

    def update_classification_learning_rate(self):
        step = self.args.class_lr / self.args.class_round
        self.classification_learning_rate -= step
        pass
