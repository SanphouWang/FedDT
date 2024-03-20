from pathlib import Path
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

    def download_model(self):
        # Download the latest model from the server
        pass

    def train_gen_model(self, client_idx):
        # Train the generation model locally
        train_loader = DataLoader(
            self.dataset_list[client_idx], batch_size=self.args.gen_batch_size, shuffle=True
        )
        gen_model = get_model_arch(self.args.gen_model)(self.args, self.device, train_loader)
        for i in range(self.args.gen_epochs):
            gen_model.train_epoch()
            print(f"client {client_idx} epoch {i} finished")
        return gen_model.get_weigts()

    def upload_model(self):
        # Upload the trained model to the server
        pass

    def get_training_data(self):
        # Get the training data from a local data source
        pass
