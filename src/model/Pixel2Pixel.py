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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())

from src.model.basic_models import (
    CycleDis,
    CycleGen,
    ResnetGenerator,
    NLayerDiscriminator,
    UnetGenerator,
    MLPDiscriminator,
    MLPGenerator,
    Pixel2PixelDiscriminator,
)
from src.model.big_unet_generator import BigUNetGenerator
from src.utils.tools import move2cpu, move2device
from src.utils.metrics import METRIC_MAPPING
from src.model.CycleGAN import CycleGAN, GANLoss

DATASET_MODEL_MAP = {
    "brats2019": {
        "generator": ResnetGenerator(n_blocks=9),
        "discriminator": NLayerDiscriminator(n_layers=6),
    },
    "adni_roi": {
        "generator": MLPGenerator(),
        "discriminator": Pixel2PixelDiscriminator(),
    },
}


class Pixel2Pixel(CycleGAN):
    def __init__(
        self,
        args,
        logger,
        device=None,
        mode="train",
    ):
        self.args = args
        self.device = device
        self.mode = mode
        self.logger = logger
        self.gen_021 = DATASET_MODEL_MAP[self.args.dataset]["generator"]
        self.gen_120 = DATASET_MODEL_MAP[self.args.dataset]["generator"]
        self.generator_list = [self.gen_120, self.gen_021]
        self.generate_image_flag = False  # for fix batch for generating image
        self.current_round = 0  # record current step for evaluating training process
        if mode == "train":
            self.initialize_training_component()

    def initialize_training_component(self):
        self.dis_0 = DATASET_MODEL_MAP[self.args.dataset]["discriminator"]
        self.dis_1 = DATASET_MODEL_MAP[self.args.dataset]["discriminator"]
        self.dis_list = [self.dis_0, self.dis_1]
        self.optimizer_gen_021 = torch.optim.SGD(
            self.gen_021.parameters(),
            lr=self.args.gen_lr,
            momentum=0.9,
        )
        self.optimizer_gen_120 = torch.optim.SGD(
            self.gen_120.parameters(),
            lr=self.args.gen_lr,
            momentum=0.9,
        )
        self.optimizer_dis_0 = torch.optim.SGD(
            self.dis_0.parameters(),
            lr=self.args.dis_lr,
            momentum=0.9,
        )
        self.optimizer_dis_1 = torch.optim.SGD(
            self.dis_1.parameters(),
            lr=self.args.dis_lr,
            momentum=0.9,
        )

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - self.args.gen_samelr_round) / float(
                self.args.gen_round - self.args.gen_samelr_round + 1
            )
            return lr_l

        self.generator_optimizer_list = [self.optimizer_gen_021, self.optimizer_gen_120]
        self.dis_optimizer_list = [self.optimizer_dis_0, self.optimizer_dis_1]
        self.shcheduler = [
            LambdaLR(self.optimizer_gen_021, lr_lambda=lambda_rule),
            LambdaLR(self.optimizer_gen_120, lr_lambda=lambda_rule),
            LambdaLR(self.optimizer_dis_0, lr_lambda=lambda_rule),
            LambdaLR(self.optimizer_dis_1, lr_lambda=lambda_rule),
        ]

    def train_epoch(self, dataloader):
        loss_generation_cache = 0.0
        loss_discrimination_cache = 0.0
        """
        Train Generators
        """
        self.post_organize = dataloader.dataset.post_organize
        # evaluate generator before training
        if (
            self.args.gen_eval_train != 0
            and (self.current_round + 1) % self.args.gen_eval_train == 0
        ):
            gen_before_trian = self.eval_generator(dataloader)
            self.logger.log(f"Generator Before Training")
            self.logger.log(
                " ".join(
                    [f"{key}: {float(value):.5f}. " for key, value in gen_before_trian.items()]
                )
            )

        self.gen_021.train()
        self.gen_120.train()
        self.dis_0.eval()
        self.dis_1.eval()

        for batch_idx, batch in enumerate(dataloader):
            loss_generation = self.gen_loss(batch)
            loss_generation_cache += loss_generation.item()
            self.optimizer_gen_021.zero_grad()
            self.optimizer_gen_120.zero_grad()
            loss_generation.backward()
            self.optimizer_gen_021.step()
            self.optimizer_gen_120.step()

        # evaluate generator after training and discriminator before training
        if (
            self.args.gen_eval_train != 0
            and (self.current_round + 1) % self.args.gen_eval_train == 0
        ):

            gen_after_train = self.eval_generator(dataloader)
            self.logger.log("Generator After Training")
            self.logger.log(
                " ".join([f"{key}: {float(value):.5f}. " for key, value in gen_after_train.items()])
            )
            dis_before_train = self.eval_discriminator(dataloader)
            self.logger.log("Discriminator Before Training")
            self.logger.log(
                " ".join(
                    [f"{key}: {float(value):.5f}. " for key, value in dis_before_train.items()]
                )
            )

        """
        Train Discriminator
        """
        self.gen_021.eval()
        self.gen_120.eval()
        self.dis_0.train()
        self.dis_1.train()
        if (self.current_round + 1) % self.args.dis_train_gap == 0:
            for batch_idx, batch in enumerate(dataloader):
                loss_discrimination = self.dis_loss(batch)
                loss_discrimination_cache += loss_discrimination.item()
                self.optimizer_dis_0.zero_grad()
                self.optimizer_dis_1.zero_grad()
                loss_discrimination.backward()
                self.optimizer_dis_0.step()
                self.optimizer_dis_1.step()
            loss_discrimination_cache = loss_discrimination_cache / len(dataloader)
            loss_generation_cache = loss_generation_cache / len(dataloader)
            # evaluate discriminator after training
            if (
                self.args.gen_eval_train != 0
                and (self.current_round + 1) % self.args.gen_eval_train == 0
            ):

                dis_after_train = self.eval_discriminator(dataloader)
                self.logger.log("Discriminator After Training")
                self.logger.log(
                    " ".join(
                        [f"{key}: {float(value):.5f}. " for key, value in dis_after_train.items()]
                    )
                )
        # self.current_round += 1
        return {
            "generation_loss": loss_generation_cache,
            "discrimination_loss": loss_discrimination_cache,
        }

    def gen_loss(self, batch, evaluate_training=False):
        """
        calculate loss and
        """
        criterion_GAN = GANLoss().to(self.device)
        criterion_pixel = nn.L1Loss().to(self.device)
        batch, modality0_indices, modality1_indices, _, _ = batch
        complete_indices = list(set(modality0_indices) & set(modality1_indices))
        loss_GAN_021 = torch.tensor(0.0, dtype=torch.float)
        loss_GAN_120 = torch.tensor(0.0, dtype=torch.float)
        loss_pixel = torch.tensor(0.0, dtype=torch.float)
        loss_generation = torch.tensor(0.0, dtype=torch.float)
        if len(complete_indices) > 0:
            modality0_volumes = batch["modality0"][complete_indices]
            modality1_volumes = batch["modality1"][complete_indices]
            real_modality0 = modality0_volumes.to(self.device)
            real_modality1 = modality1_volumes.to(self.device)
            # real_modality0 = self.post_organize(modality0_volumes).to(self.device)
            # real_modality1 = self.post_organize(modality1_volumes).to(self.device)
            fake_modality1 = self.gen_021(real_modality0)
            fake_modality0 = self.gen_120(real_modality1)
            # GAN loss
            pred_fake_021 = self.dis_1(real_modality0, fake_modality1)
            # pred_fake_021 = self.dis_1(fake_modality1)
            # pred_fake_021 = self.dis_1(torch.cat((real_modality1, fake_modality1), 1))
            loss_GAN_021 = criterion_GAN(pred_fake_021, True)
            # pred_fake_120 = self.dis_0(torch.cat((real_modality0, fake_modality0), 1))
            pred_fake_120 = self.dis_0(real_modality1, fake_modality0)
            loss_GAN_120 = criterion_GAN(pred_fake_120, True)
            loss_GAN = loss_GAN_021 + loss_GAN_120
            # pixel loss
            loss_pixel_021 = criterion_pixel(fake_modality1, real_modality1)
            loss_pixel_120 = criterion_pixel(fake_modality0, real_modality0)
            loss_pixel = loss_pixel_021 + loss_pixel_120
            loss_generation = self.args.lambda_gan * loss_GAN + self.args.lambda_pixel * loss_pixel
        return_dict = {
            "gan_loss": self.args.lambda_gan * (loss_GAN_021 + loss_GAN_120),
            "pixel_loss": self.args.lambda_pixel * loss_pixel,
        }
        if evaluate_training:
            return return_dict
        else:
            return loss_generation

    def dis_loss(self, batch):
        criterion_GAN = GANLoss().to(self.device)
        batch, modality0_indices, modality1_indices, _, _ = batch
        loss_discrimination = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        complete_indices = list(set(modality0_indices) & set(modality1_indices))
        modality0_volumes = batch["modality0"][complete_indices]
        modality1_volumes = batch["modality1"][complete_indices]
        if len(complete_indices) > 0:
            real_modality0 = self.post_organize(modality0_volumes).to(self.device)
            real_modality1 = self.post_organize(modality1_volumes).to(self.device)
            fake_modality1 = self.gen_021(real_modality0)
            fake_modality0 = self.gen_120(real_modality1)
            pred_fake1 = self.dis_1(real_modality0, fake_modality1)
            pred_fake0 = self.dis_0(real_modality1, fake_modality0)
            pred_real0 = self.dis_0(real_modality1, real_modality0)
            pred_real1 = self.dis_1(real_modality0, real_modality1)
            loss_discrimination = (
                criterion_GAN(pred_real0, True)
                + criterion_GAN(pred_real1, True)
                + criterion_GAN(pred_fake0, False)
                + criterion_GAN(pred_fake1, False)
            )

        return loss_discrimination
