from collections import OrderedDict
from pathlib import Path
from typing import List
import torch

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
from data.utils.datasets import DATASETS


from src.utils.models import get_model_arch


class FedAvgClient:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.client_num = self.args.client_num
        self.transform = ToTensor()
        self.dataset_list = [
            DATASETS[self.args.dataset](self.args, client_id=i) for i in range(self.client_num)
        ]
        # initialize model for generation
        self.train_loader_list = [
            DataLoader(self.dataset_list[i], batch_size=self.args.gen_batch_size, shuffle=True)
            for i in range(self.client_num)
        ]
        # gen model refer to generation model class, e.g. CycleGAN class
        self.gen_model_list = [
            get_model_arch(self.args.gen_model)(self.args, self.device, train_loader)
            for train_loader in self.train_loader_list
        ]

    def get_client_weights(self) -> List[float]:
        # generate weight of each client for model aggregation in server
        dataset_len_list = [len(dataset) for dataset in self.dataset_list]
        total_num = sum(dataset_len_list)
        weights_list = [len_dataset / total_num for len_dataset in dataset_len_list]
        return weights_list

    def get_model_weights(self) -> List[List[OrderedDict]]:
        # return weights of models on each client
        # since each model may have different submodels (e.g. CycleGAN has two generator and discriminator)
        # each element in the returned list is a list of weights of submodels (e.g. [weights of generator1, weights of generator2, ...])
        return [model.get_weights() for model in self.gen_model_list]

    def download_model(self, aggregated_model_dict_list):
        for model in self.gen_model_list:
            model.download_weights(aggregated_model_dict_list)
        pass

    def train_gen_model(self, client_idx):
        # Train the generation model locally
        self.gen_model_list[client_idx].move2device()
        for i in range(self.args.gen_epochs):
            self.gen_model_list[client_idx].train_epoch()
            print(f"client {client_idx} epoch {i} finished")
        self.gen_model_list[client_idx].move2cpu()

    def upload_model(self):
        # Upload the trained model to the server
        pass

    def get_training_data(self):
        # Get the training data from a local data source
        pass
