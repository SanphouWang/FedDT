from collections import OrderedDict
from pathlib import Path
import sys
from typing import List
from torch.autograd import Variable

import torch

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())

from src.utils.basic_models import CycleDis, CycleGen


def get_model_arch(model_name):
    # static means the model arch is fixed.
    static = {
        "cyclegan": CycleGAN,
    }
    if model_name in static:
        return static[model_name]
    else:
        raise ValueError(f"Unsupported model: {model_name}")


class CycleGAN:
    def __init__(
        self,
        args,
        device,
        trainloader,
    ) -> None:
        self.args = args
        self.device = device
        self.trainloader = trainloader
        # generators and discriminators
        self.cyclegen_021 = CycleGen()  # generate from modality0 to modality1
        self.cyclegen_120 = CycleGen()  # generate from modality1 to modality0
        self.cycledis_0 = CycleDis()  # discriminate fake modality0 from true image
        self.cycledis_1 = CycleDis()  # discriminate fake modality1 from true image
        # optimizers
        self.optimizer_cyclegen_021 = torch.optim.Adam(
            self.cyclegen_021.parameters(),
            lr=self.args.gen_lr,
            betas=(self.args.gen_beta1, self.args.gen_beta2),
        )
        self.optimizer_cyclegen_120 = torch.optim.Adam(
            self.cyclegen_120.parameters(),
            lr=self.args.gen_lr,
            betas=(self.args.gen_beta1, self.args.gen_beta2),
        )
        self.optimizer_cycledis_0 = torch.optim.Adam(
            self.cycledis_0.parameters(),
            lr=self.args.gen_lr,
            betas=(self.args.gen_beta1, self.args.gen_beta2),
        )
        self.optimizer_cycledis_1 = torch.optim.Adam(
            self.cycledis_1.parameters(),
            lr=self.args.gen_lr,
            betas=(self.args.gen_beta1, self.args.gen_beta2),
        )

    def train_epoch(self):
        criterion_GAN = torch.nn.MSELoss().to(self.device)
        criterion_cycle = torch.nn.L1Loss().to(self.device)
        criterion_identity = torch.nn.L1Loss().to(self.device)
        for batch_idx, batch in enumerate(self.trainloader):
            volumes = batch[0]
            modalities: List[str] = batch[1]
            modality0_indices = [
                i for i, modality in enumerate(modalities) if modality == self.args.modality0
            ]  # indices of modality0 in modalities
            modality1_indices = [
                i for i, modality in enumerate(modalities) if modality == self.args.modality1
            ]
            """
            Train Generators
            """
            self.cyclegen_021.train()
            self.cyclegen_120.train()
            self.cycledis_1.eval()
            self.cycledis_0.eval()
            # initialize loss terms
            loss_GAN_021 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            loss_id_021 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            loss_cycle_021 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            loss_GAN_120 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            loss_id_120 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            loss_cycle_120 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            # if this batch has modality 0
            if len(modality0_indices) > 0:
                modality0_volumes = volumes[modality0_indices]
                real_modality0 = modality0_volumes.view(
                    -1, 1, volumes.shape[2], volumes.shape[3]
                ).to(self.device)
                valid = Variable(
                    torch.ones((real_modality0.size(0), 1)),
                    requires_grad=False,
                ).to(self.device)
                fake_modality1 = self.cyclegen_021(real_modality0)
                loss_GAN_021 = criterion_GAN(self.cycledis_1(fake_modality1), valid)
                loss_id_021 = criterion_identity(self.cyclegen_120(real_modality0), real_modality0)
                loss_cycle_021 = criterion_cycle(self.cyclegen_120(fake_modality1), real_modality0)
            # if this batch has modality 1
            if len(modality1_indices) > 0:
                modality1_volumes = volumes[modality1_indices]
                real_modality1 = modality1_volumes.view(
                    -1, 1, volumes.shape[2], volumes.shape[3]
                ).to(self.device)
                valid = Variable(
                    torch.ones((real_modality1.size(0), 1)),
                    requires_grad=False,
                ).to(self.device)
                fake_modality0 = self.cyclegen_120(real_modality1)
                loss_GAN_120 = criterion_GAN(self.cycledis_0(fake_modality0), valid)
                loss_id_120 = criterion_identity(self.cyclegen_021(real_modality1), real_modality1)
                loss_cycle_120 = criterion_cycle(self.cyclegen_021(fake_modality0), real_modality1)
            loss_generation = (
                self.args.lambda_gan * (loss_GAN_021 + loss_GAN_120)
                + self.args.lambda_identity * (loss_id_021 + loss_id_120)
                + self.args.lambda_cycle * (loss_cycle_021 + loss_cycle_120)
            )
            self.optimizer_cyclegen_021.zero_grad()
            self.optimizer_cyclegen_120.zero_grad()
            loss_generation.backward()
            self.optimizer_cyclegen_021.step()
            self.optimizer_cyclegen_120.step()

            """
            Train Discriminator
            """
            self.cycledis_0.train()
            self.cycledis_1.train()
            loss_fake_1 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            loss_real_0 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            loss_fake_0 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            loss_real_1 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            # if this batch has modality 0
            if len(modality0_indices) > 0:
                fake_modality1 = fake_modality1.detach()
                valid = Variable(
                    torch.ones((real_modality0.size(0), 1)),
                    requires_grad=False,
                ).to(self.device)
                fake = Variable(
                    torch.zeros((real_modality0.size(0), 1)),
                    requires_grad=False,
                ).to(self.device)
                loss_fake_1 = criterion_GAN(self.cycledis_1(fake_modality1), fake)
                loss_real_0 = criterion_GAN(self.cycledis_0(real_modality0), valid)
            # if this batch has modality 1
            if len(modality1_indices) > 0:
                fake_modality0 = fake_modality0.detach()
                valid = Variable(
                    torch.ones((real_modality1.size(0), 1)),
                    requires_grad=False,
                ).to(self.device)
                fake = Variable(
                    torch.zeros((real_modality1.size(0), 1)),
                    requires_grad=False,
                ).to(self.device)
                loss_fake_0 = criterion_GAN(self.cycledis_0(fake_modality0), fake)
                loss_real_1 = criterion_GAN(self.cycledis_1(real_modality1), valid)
            loss_discrimination = loss_fake_0 + loss_fake_1 + loss_real_0 + loss_real_1
            self.optimizer_cycledis_0.zero_grad()
            self.optimizer_cycledis_1.zero_grad()
            loss_discrimination.backward()
            self.optimizer_cycledis_0.step()
            self.optimizer_cycledis_1.step()
            print(f"batch {batch_idx} finished")

    def move2cpu(self):
        self.cycledis_0.to("cpu")
        self.cycledis_1.to("cpu")
        self.cyclegen_120.to("cpu")
        self.cyclegen_021.to("cpu")

    def move2device(self):
        self.cyclegen_021.to(self.device)
        self.cyclegen_120.to(self.device)
        self.cycledis_0.to(self.device)
        self.cycledis_1.to(self.device)

    def get_weights(self) -> List[OrderedDict]:
        return [
            self.cyclegen_120.state_dict(),
            self.cyclegen_021.state_dict(),
        ]

    def download_weights(self, weights: List[OrderedDict]):
        self.cyclegen_021.load_state_dict(weights[0])
        self.cyclegen_120.load_state_dict(weights[1])


if __name__ == "__main__":
    pass
