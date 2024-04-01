from collections import OrderedDict
import os
from pathlib import Path
import random
import sys
from typing import Dict, List
from torch.autograd import Variable
import matplotlib.pyplot as plt

import torch
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.regression import MeanAbsoluteError
import matplotlib.pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())

from src.utils.basic_models import (
    CycleDis,
    CycleGen,
    ResnetGenerator,
    NLayerDiscriminator,
    UnetGenerator,
)

from src.utils.tools import move2cpu, move2device


class CycleGAN:
    def __init__(self, args, device, dataloader, mode="train") -> None:
        self.args = args
        self.device = device
        self.dataloader = dataloader
        self.mode = mode
        # generators, discriminators and optimizers
        # self.cyclegen_021 = CycleGen()  # generate from modality0 to modality1
        # self.cyclegen_120 = CycleGen()  # generate from modality1 to modality0
        # self.cyclegen_021 = ResnetGenerator(1, 1)
        # self.cyclegen_120 = ResnetGenerator(1, 1)
        self.cyclegen_021 = UnetGenerator()
        self.cyclegen_120 = UnetGenerator()
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

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - 20) / float(self.args.gen_round - 20 + 1)
            return lr_l

        self.shcheduler = [
            LambdaLR(self.optimizer_cyclegen_021, lr_lambda=lambda_rule),
            LambdaLR(self.optimizer_cyclegen_120, lr_lambda=lambda_rule),
        ]
        if mode == "train":
            # self.cycledis_0 = CycleDis(
            #     input_shape=self.dataloader.dataset.image_shape
            # )  # discriminate fake modality0 from true image
            # self.cycledis_1 = CycleDis(
            #     input_shape=self.dataloader.dataset.image_shape
            # )  # discriminate fake modality1 from true image
            self.cycledis_0 = NLayerDiscriminator()
            self.cycledis_1 = NLayerDiscriminator()
            self.optimizer_cycledis_0 = torch.optim.Adam(
                self.cycledis_0.parameters(),
                lr=self.args.dis_lr,
                betas=(self.args.gen_beta1, self.args.gen_beta2),
            )
            self.optimizer_cycledis_1 = torch.optim.Adam(
                self.cycledis_1.parameters(),
                lr=self.args.dis_lr,
                betas=(self.args.gen_beta1, self.args.gen_beta2),
            )
            self.shcheduler += [
                LambdaLR(self.optimizer_cycledis_0, lr_lambda=lambda_rule),
                LambdaLR(self.optimizer_cycledis_1, lr_lambda=lambda_rule),
            ]
            # self.output_shape = self.cycledis_0.output_shape
        self.dis_train_epoch = 0

    def generate_indices(self, batch):
        return self.dataloader.dataset.generate_indices(self.args, batch)

    def evaluate_training(self):
        self.cyclegen_021.eval()
        self.cyclegen_120.eval()
        self.cycledis_0.eval()
        self.cycledis_1.eval()
        loss_generation_cache = {}
        loss_discrimination_cache = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                loss_generation: Dict = self.gen_loss(batch, evaluate_training=True)
                loss_discrimination = self.dis_loss(batch)
                if batch_idx == 0:
                    for key in loss_generation.keys():
                        loss_generation_cache[key] = loss_generation[key]
                else:
                    for key in loss_generation.keys():
                        loss_generation_cache[key] += loss_generation[key]
                loss_discrimination_cache += loss_discrimination
        for key in loss_generation_cache.keys():
            loss_generation_cache[key] = loss_generation_cache[key] / len(self.dataloader)
        return_dict = loss_generation_cache
        return_dict["discrimination_loss"] = loss_discrimination_cache / len(self.dataloader)
        return return_dict

    def update_learning_rate(self):
        for scheduler in self.shcheduler:
            scheduler.step()

    def train_epoch(self):
        loss_generation_cache = 0.0
        loss_discrimination_cache = 0.0

        """
        Train Generators
        """
        for batch_idx, batch in enumerate(self.dataloader):
            self.cyclegen_021.train()
            self.cyclegen_120.train()
            self.cycledis_0.eval()
            self.cycledis_1.eval()
            loss_generation = self.gen_loss(batch)
            loss_generation_cache += loss_generation.item()
            self.optimizer_cyclegen_021.zero_grad()
            self.optimizer_cyclegen_120.zero_grad()
            loss_generation.backward()
            self.optimizer_cyclegen_021.step()
            self.optimizer_cyclegen_120.step()

            """
            Train Discriminator
            """
        if self.dis_train_epoch % self.args.dis_train_gap == 0:
            for batch_idx, batch in enumerate(self.dataloader):
                self.cyclegen_021.eval()
                self.cyclegen_120.eval()
                self.cycledis_0.train()
                self.cycledis_1.train()
                loss_discrimination = self.dis_loss(batch)
                loss_discrimination_cache += loss_discrimination.item()
                self.optimizer_cycledis_0.zero_grad()
                self.optimizer_cycledis_1.zero_grad()
                loss_discrimination.backward()
                self.optimizer_cycledis_0.step()
                self.optimizer_cycledis_1.step()
            loss_discrimination_cache = loss_discrimination_cache / len(self.dataloader)
            loss_generation_cache = loss_generation_cache / len(self.dataloader)
        self.dis_train_epoch += 1
        return {
            "generation_loss": loss_generation_cache,
            "discrimination_loss": loss_discrimination_cache,
        }

    def gen_loss(self, batch, evaluate_training=False):
        criterion_GAN = GANLoss().to(self.device)
        criterion_cycle = torch.nn.L1Loss().to(self.device)
        criterion_identity = torch.nn.L1Loss().to(self.device)
        criterion_paired = torch.nn.L1Loss().to(self.device)
        modality0_indices, modality1_indices, paired_in_modality0, paired_in_modality1 = (
            self.generate_indices(batch)
        )
        modality0_volumes = batch["modality0"]
        modality1_volumes = batch["modality1"]

        # initialize loss terms
        loss_GAN_021 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        loss_id_021 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        loss_cycle_021 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        loss_GAN_120 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        loss_id_120 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        loss_cycle_120 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        # if this batch has modality 0
        if len(modality0_indices) > 0:
            modality0_volumes = modality0_volumes[modality0_indices]
            real_modality0 = modality0_volumes.view(
                -1, 1, modality0_volumes.shape[2], modality0_volumes.shape[3]
            ).to(self.device)
            fake_modality1 = self.cyclegen_021(real_modality0)
            loss_GAN_021 = criterion_GAN(self.cycledis_1(fake_modality1), True)
            loss_id_021 = criterion_identity(self.cyclegen_120(real_modality0), real_modality0)
            loss_cycle_021 = criterion_cycle(self.cyclegen_120(fake_modality1), real_modality0)
        # if this batch has modality 1
        if len(modality1_indices) > 0:
            modality1_volumes = modality1_volumes[modality1_indices]
            real_modality1 = modality1_volumes.view(
                -1, 1, modality1_volumes.shape[2], modality1_volumes.shape[3]
            ).to(self.device)
            fake_modality0 = self.cyclegen_120(real_modality1)
            loss_GAN_120 = criterion_GAN(self.cycledis_0(fake_modality0), True)
            loss_id_120 = criterion_identity(self.cyclegen_021(real_modality1), real_modality1)
            loss_cycle_120 = criterion_cycle(self.cyclegen_021(fake_modality0), real_modality1)
        loss_generation = (
            self.args.lambda_gan * (loss_GAN_021 + loss_GAN_120)
            + self.args.lambda_identity * (loss_id_021 + loss_id_120)
            + self.args.lambda_cycle * (loss_cycle_021 + loss_cycle_120)
        )
        return_dict = {
            "gan_loss": self.args.lambda_gan * (loss_GAN_021 + loss_GAN_120),
            "identity_loss": self.args.lambda_identity * (loss_id_021 + loss_id_120),
            "cycle_loss": self.args.lambda_cycle * (loss_cycle_021 + loss_cycle_120),
        }
        if (
            len(paired_in_modality0) > 0
            and len(paired_in_modality1) > 0
            and self.args.lambda_paired > 0
        ):
            real_paired_modality0 = real_modality0[paired_in_modality0]
            real_paired_modality1 = real_modality1[paired_in_modality1]
            fake_paired_modality0 = fake_modality0[paired_in_modality0]
            fake_paired_modality1 = fake_modality1[paired_in_modality1]
            loss_paired_0 = criterion_paired(fake_paired_modality0, real_paired_modality0)
            loss_paired_1 = criterion_paired(fake_paired_modality1, real_paired_modality1)
            loss_generation += self.args.lambda_paired * (loss_paired_0 + loss_paired_1)
            return_dict["paired_loss"] = self.args.lambda_paired * (loss_paired_0 + loss_paired_1)
        if evaluate_training:
            return return_dict
        else:
            return loss_generation

    def dis_loss(
        self,
        batch,
    ):

        loss_fake_1 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        loss_real_0 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        loss_fake_0 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        loss_real_1 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        criterion_GAN = GANLoss().to(self.device)
        modality0_indices, modality1_indices, paired_in_modality0, paired_in_modality1 = (
            self.generate_indices(batch)
        )
        modality0_volumes = batch["modality0"]
        modality1_volumes = batch["modality1"]
        if len(modality0_indices) > 0:
            real_modality0 = modality0_volumes.view(
                -1, 1, modality0_volumes.shape[2], modality0_volumes.shape[3]
            ).to(self.device)
            fake_modality1 = self.cyclegen_021(real_modality0)
            # fake_modality1 = fake_modality1
            # valid = Variable(
            #     torch.ones((real_modality0.size(0), *self.output_shape)),
            #     requires_grad=False,
            # ).to(self.device)
            # fake = Variable(
            #     torch.zeros((real_modality0.size(0), *self.output_shape)),
            #     requires_grad=False,
            # ).to(self.device)
            loss_fake_1 = criterion_GAN(self.cycledis_1(fake_modality1), False)
            loss_real_0 = criterion_GAN(self.cycledis_0(real_modality0), True)
        # if this batch has modality 1
        if len(modality1_indices) > 0:
            real_modality1 = modality1_volumes.view(
                -1, 1, modality1_volumes.shape[2], modality1_volumes.shape[3]
            ).to(self.device)
            # fake_modality0 = fake_modality0
            fake_modality0 = self.cyclegen_120(real_modality1)
            # valid = Variable(
            #     torch.ones((real_modality1.size(0), *self.output_shape)),
            #     requires_grad=False,
            # ).to(self.device)
            # fake = Variable(
            #     torch.zeros((real_modality1.size(0), *self.output_shape)),
            #     requires_grad=False,
            # ).to(self.device)
            loss_fake_0 = criterion_GAN(self.cycledis_0(fake_modality0), False)
            loss_real_1 = criterion_GAN(self.cycledis_1(real_modality1), True)
        loss_discrimination = loss_fake_0 + loss_fake_1 + loss_real_0 + loss_real_1
        return loss_discrimination

    def move2cpu(self):
        # if isinstance(self.cyclegen_021, torch.nn.DataParallel):
        #     self.cyclegen_021 = self.cyclegen_021.module
        # if isinstance(self.cyclegen_120, torch.nn.DataParallel):
        #     self.cyclegen_120 = self.cyclegen_120.module
        # self.cyclegen_120.cpu()
        # self.cyclegen_021.cpu()
        # if self.mode == "train":
        #     if isinstance(self.cycledis_0, torch.nn.DataParallel):
        #         self.cycledis_0 = self.cycledis_0.module
        #     if isinstance(self.cycledis_1, torch.nn.DataParallel):
        #         self.cycledis_1 = self.cycledis_1.module
        #     self.cycledis_0.cpu()
        #     self.cycledis_1.cpu()
        self.cyclegen_021 = move2cpu(self.cyclegen_021)
        self.cyclegen_120 = move2cpu(self.cyclegen_120)
        if self.mode == "train":
            self.cycledis_0 = move2cpu(self.cycledis_0)
            self.cycledis_1 = move2cpu(self.cycledis_1)

    def move2device(self):
        self.cyclegen_021 = move2device(self.device, self.args.multi_gpu, self.cyclegen_021)
        self.cyclegen_120 = move2device(self.device, self.args.multi_gpu, self.cyclegen_120)
        if self.mode == "train":
            self.cycledis_0 = move2device(self.device, self.args.multi_gpu, self.cycledis_0)
            self.cycledis_1 = move2device(self.device, self.args.multi_gpu, self.cycledis_1)

    def get_weights(self, check_point=False) -> List[OrderedDict]:
        return [
            self.cyclegen_120.state_dict(),
            self.cyclegen_021.state_dict(),
            self.cycledis_0.state_dict(),
            self.cycledis_1.state_dict(),
        ]

    def download_weights(self, weights: List[OrderedDict], resume=False):
        self.cyclegen_120.load_state_dict(weights[0])
        self.cyclegen_021.load_state_dict(weights[1])
        if (self.args.upload_dis and self.mode == "train") or resume:
            self.cycledis_0.load_state_dict(weights[2])
            self.cycledis_1.load_state_dict(weights[3])

    def get_generator(self):
        return self.cyclegen_021, self.cyclegen_120

    def test(
        self,
    ):
        self.cyclegen_021.eval()
        self.cyclegen_120.eval()
        psnr_func = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_func = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        mae_func = MeanAbsoluteError().to(self.device)
        psnr_021_list = []
        psnr_120_list = []
        ssim_021_list = []
        ssim_120_list = []
        mae_021_list = []
        mae_120_list = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                real_modality0 = batch["modality0"]
                real_modality1 = batch["modality1"]
                real_modality0 = real_modality0.view(
                    -1, 1, real_modality0.shape[2], real_modality0.shape[3]
                ).to(self.device)
                real_modality1 = real_modality1.view(
                    -1, 1, real_modality1.shape[2], real_modality1.shape[3]
                ).to(self.device)
                fake_modality1 = self.cyclegen_021(real_modality0)
                fake_modality0 = self.cyclegen_120(real_modality1)
                # calculate psnr and ssim
                psnr_021_list.append(psnr_func(fake_modality1, real_modality1))
                psnr_120_list.append(psnr_func(fake_modality0, real_modality0))
                ssim_021_list.append(ssim_func(fake_modality1, real_modality1))
                ssim_120_list.append(ssim_func(fake_modality0, real_modality0))
                mae_021_list.append(mae_func(fake_modality1, real_modality1))
                mae_120_list.append(mae_func(fake_modality0, real_modality0))
        mean_psnr_021 = torch.mean(torch.stack(psnr_021_list), dim=0).cpu()
        mean_psnr_120 = torch.mean(torch.stack(psnr_120_list), dim=0).cpu()
        mean_ssim_021 = torch.mean(torch.stack(ssim_021_list), dim=0).cpu()
        mean_ssim_120 = torch.mean(torch.stack(ssim_120_list), dim=0).cpu()
        mean_mae_021 = torch.mean(torch.stack(mae_021_list), dim=0).cpu()
        mean_mae_120 = torch.mean(torch.stack(mae_120_list), dim=0).cpu()
        return {
            "psnr_021": round(mean_psnr_021.item(), 4),
            "psnr_120": round(mean_psnr_120.item(), 4),
            "ssim_021": round(mean_ssim_021.item(), 4),
            "ssim_120": round(mean_ssim_120.item(), 4),
            "mae_021": round(mean_mae_021.item(), 4),
            "mae_120": round(mean_mae_120.item(), 4),
        }

    def plot_test_result(self, test_result_list, test_rounds, out_dir):
        cat_metric = {}
        test_rounds = [int(round) for round in test_rounds]
        for metric_name in test_result_list[0].keys():
            cat_metric[metric_name] = []
        for metric_at_round in test_result_list:
            for key, value in metric_at_round.items():
                cat_metric[key].append(value)
        for metric_name in ["psnr", "ssim", "mae"]:
            plt.figure()
            plt.plot(
                test_rounds,
                cat_metric[f"{metric_name}_021"],
                label=f"{self.args.modality0.upper()} -> {self.args.modality1.upper()}",
            )
            plt.plot(
                test_rounds,
                cat_metric[f"{metric_name}_120"],
                label=f"{self.args.modality1.upper()} -> {self.args.modality0.upper()}",
            )

            plt.xlabel("Test Round")
            # plt.ylabel(metric_name)
            plt.legend()
            plt.title(metric_name.upper())
            plt.savefig(f"{out_dir}/{metric_name}.png")

    def generate_image(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self.cyclegen_021.eval()
        self.cyclegen_120.eval()
        with torch.no_grad():

            batch = next(iter(self.dataloader))
            real_modality0 = batch["modality0"].to(self.device)
            real_modality1 = batch["modality1"].to(self.device)
            real_modality0 = real_modality0.view(
                -1, 1, real_modality0.shape[2], real_modality0.shape[3]
            ).to(self.device)
            real_modality1 = real_modality1.view(
                -1, 1, real_modality1.shape[2], real_modality1.shape[3]
            ).to(self.device)
            fake_modality1 = self.cyclegen_021(real_modality0)
            fake_modality0 = self.cyclegen_120(real_modality1)
            # randomly select 4 slices
            slice_idx = random.sample(range(real_modality0.shape[0]), 4)
            real_modality0 = real_modality0[slice_idx]
            real_modality1 = real_modality1[slice_idx]
            fake_modality1 = fake_modality1[slice_idx]
            fake_modality0 = fake_modality0[slice_idx]
            for i in range(4):
                fig, axs = plt.subplots(1, 4, figsize=(12, 6))
                # for modality in [real_modality0, real_modality1, fake_modality1, fake_modality0]:
                axs[0].imshow(real_modality0[i].squeeze().cpu().numpy(), cmap="gray")
                axs[0].set_title(f"Real {self.args.modality0.upper()}")
                axs[0].axis("off")
                axs[1].imshow(real_modality1[i].squeeze().cpu().numpy(), cmap="gray")
                axs[1].set_title(f"Real {self.args.modality1.upper()}")
                axs[1].axis("off")
                axs[2].imshow(fake_modality1[i].squeeze().cpu().numpy(), cmap="gray")
                axs[2].set_title(f"Fake {self.args.modality1.upper()}")
                axs[2].axis("off")
                axs[3].imshow(fake_modality0[i].squeeze().cpu().numpy(), cmap="gray")
                axs[3].set_title(f"Fake {self.args.modality0.upper()}")
                axs[3].axis("off")
                plt.tight_layout()

                plt.savefig(f"{out_dir}/generated{i}.jpg")


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """

        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)

        return loss


if __name__ == "__main__":
    pass
