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


class CGAN_P2P(CycleGAN):
    def __init__(
        self,
        args,
        logger,
        device=None,
        mode="train",
    ):
        super().__init__(args, logger, device, mode)
        assert self.args.lambda_paired > 0

    def initialize_training_component(self):
        self.dis_0_cycle = MLPDiscriminator()
        self.dis_1_cycle = MLPDiscriminator()
        self.dis_0_p2p = Pixel2PixelDiscriminator()
        self.dis_1_p2p = Pixel2PixelDiscriminator()
        self.dis_list = [self.dis_0_cycle, self.dis_1_cycle, self.dis_0_p2p, self.dis_1_p2p]
        self.optimizer_gen_021 = torch.optim.Adam(
            self.gen_021.parameters(),
            lr=self.args.gen_lr,
            # momentum=0.9,
            weight_decay=0.00001,
        )
        self.optimizer_gen_120 = torch.optim.Adam(
            self.gen_120.parameters(),
            lr=self.args.gen_lr,
            # momentum=0.9,
            weight_decay=0.00001,
        )
        self.optimizer_dis_0_cycle = torch.optim.Adam(
            self.dis_0_cycle.parameters(),
            lr=self.args.dis_lr,
            # momentum=0.9,
            weight_decay=0.00001,
        )
        self.optimizer_dis_1_cycle = torch.optim.Adam(
            self.dis_1_cycle.parameters(),
            lr=self.args.dis_lr,
            # momentum=0.9,
            weight_decay=0.00001,
        )
        self.optimizer_dis_0_p2p = torch.optim.Adam(
            self.dis_0_p2p.parameters(),
            lr=self.args.dis_lr,
            # momentum=0.9,
            weight_decay=0.00001,
        )
        self.optimizer_dis_1_p2p = torch.optim.Adam(
            self.dis_1_cycle.parameters(),
            lr=self.args.dis_lr,
            # momentum=0.9,
            weight_decay=0.00001,
        )
        self.generator_optimizer_list = [self.optimizer_gen_021, self.optimizer_gen_120]
        self.dis_optimizer_list = [
            self.optimizer_dis_0_cycle,
            self.optimizer_dis_1_cycle,
            self.optimizer_dis_0_p2p,
            self.optimizer_dis_1_p2p,
        ]

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - self.args.gen_samelr_round) / float(
                self.args.gen_round - self.args.gen_samelr_round + 1
            )
            return lr_l

        self.shcheduler = [
            LambdaLR(self.optimizer_gen_021, lr_lambda=lambda_rule),
            LambdaLR(self.optimizer_gen_120, lr_lambda=lambda_rule),
            LambdaLR(self.optimizer_dis_0_cycle, lr_lambda=lambda_rule),
            LambdaLR(self.optimizer_dis_1_cycle, lr_lambda=lambda_rule),
            LambdaLR(self.optimizer_dis_0_p2p, lr_lambda=lambda_rule),
            LambdaLR(self.optimizer_dis_1_p2p, lr_lambda=lambda_rule),
        ]

    def move2cpu(self):
        self.gen_021 = move2cpu(self.gen_021)
        self.gen_120 = move2cpu(self.gen_120)
        if self.mode == "train":
            self.dis_0_cycle = move2cpu(self.dis_0_cycle)
            self.dis_1_cycle = move2cpu(self.dis_1_cycle)
            self.dis_0_p2p = move2cpu(self.dis_0_p2p)
            self.dis_1_p2p = move2cpu(self.dis_1_p2p)

    def move2device(self):
        if self.device is None:
            raise ValueError("Device is not set")
        self.gen_021 = move2device(self.device, self.args.multi_gpu, self.gen_021)
        self.gen_120 = move2device(self.device, self.args.multi_gpu, self.gen_120)
        if self.mode == "train":
            self.dis_0_cycle = move2device(self.device, self.args.multi_gpu, self.dis_0_cycle)
            self.dis_1_cycle = move2device(self.device, self.args.multi_gpu, self.dis_1_cycle)
            self.dis_0_p2p = move2device(self.device, self.args.multi_gpu, self.dis_0_p2p)
            self.dis_1_p2p = move2device(self.device, self.args.multi_gpu, self.dis_1_p2p)

    def get_generator(self):
        return self.gen_021, self.gen_120

    def gen_loss(self, batch, evaluate_training=False):
        criterion_GAN = GANLoss().to(self.device)
        criterion_cycle = torch.nn.L1Loss().to(self.device)
        criterion_identity = torch.nn.L1Loss().to(self.device)
        criterion_paired = torch.nn.L1Loss().to(self.device)
        batch, modality0_indices, modality1_indices, paired_in_modality0, paired_in_modality1 = (
            batch
        )
        modality0_volumes = batch["modality0"]
        modality1_volumes = batch["modality1"]

        # initialize loss terms
        loss_GAN_021 = torch.tensor(0.0, dtype=torch.float).to(self.device)
        loss_id_021 = torch.tensor(0.0, dtype=torch.float)
        loss_cycle_021 = torch.tensor(0.0, dtype=torch.float)
        loss_GAN_120 = torch.tensor(0.0, dtype=torch.float).to(self.device)
        loss_id_120 = torch.tensor(0.0, dtype=torch.float)
        loss_cycle_120 = torch.tensor(0.0, dtype=torch.float)
        loss_paired_0 = torch.tensor(0.0, dtype=torch.float)
        loss_paired_1 = torch.tensor(0.0, dtype=torch.float)
        paired_indices = [index for index in modality0_indices if index in modality1_indices]
        single_modality1_indices = [
            index for index in modality1_indices if index not in paired_indices
        ]
        single_modality0_indices = [
            index for index in modality0_indices if index not in paired_indices
        ]
        if len(single_modality0_indices) > 0:
            single_modality0_volumes = modality0_volumes[single_modality0_indices]
            real_single_modality0 = single_modality0_volumes.to(self.device)
            fake_modality1 = self.gen_021(real_single_modality0)
            loss_GAN_021 = criterion_GAN(self.dis_1_cycle(fake_modality1), True)
            # loss_id_021 = criterion_identity(
            #     self.gen_120(real_single_modality0), real_single_modality0
            # )
            loss_cycle_021 = criterion_cycle(self.gen_120(fake_modality1), real_single_modality0)
        if len(single_modality1_indices) > 0:
            single_modality1_volumes = modality1_volumes[single_modality1_indices]
            real_single_modality1 = single_modality1_volumes.to(self.device)
            # real_modality1 = self.post_organize(modality1_volumes).to(self.device)
            fake_modality0 = self.gen_120(real_single_modality1)
            loss_GAN_120 = criterion_GAN(self.dis_0_cycle(fake_modality0), True)
            # loss_id_120 = criterion_identity(
            #     self.gen_021(real_single_modality1), real_single_modality1
            # )
            loss_cycle_120 = criterion_cycle(self.gen_021(fake_modality0), real_single_modality1)
        if len(paired_indices) > 0:
            real_paired_modality0 = modality0_volumes[paired_indices].to(self.device)
            real_paired_modality1 = modality1_volumes[paired_indices].to(self.device)
            fake_paired_modality0 = self.gen_120(real_paired_modality1)
            fake_paired_modality1 = self.gen_021(real_paired_modality0)
            loss_paired_0 = criterion_paired(fake_paired_modality0, real_paired_modality0)
            loss_paired_1 = criterion_paired(fake_paired_modality1, real_paired_modality1)
            loss_GAN_021 += criterion_GAN(
                self.dis_1_p2p(real_paired_modality0, fake_paired_modality1), True
            )
            loss_GAN_120 += criterion_GAN(
                self.dis_0_p2p(real_paired_modality1, fake_paired_modality0), True
            )
        loss_generation = (
            self.args.lambda_gan * (loss_GAN_021 + loss_GAN_120)
            # + self.args.lambda_identity * (loss_id_021 + loss_id_120)
            + self.args.lambda_cycle * (loss_cycle_021 + loss_cycle_120)
            + self.args.lambda_paired * (loss_paired_0 + loss_paired_1)
        )
        return_dict = {
            "gan_loss": self.args.lambda_gan * (loss_GAN_021 + loss_GAN_120),
            # "identity_loss": self.args.lambda_identity * (loss_id_021 + loss_id_120),
            "cycle_loss": self.args.lambda_cycle * (loss_cycle_021 + loss_cycle_120),
            "paired_loss": self.args.lambda_paired * (loss_paired_0 + loss_paired_1),
        }
        if evaluate_training:
            return return_dict
        else:
            return loss_generation

    def dis_loss(self, batch):
        loss_fake_1 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        loss_real_0 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        loss_fake_0 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        loss_real_1 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        loss_paired = torch.tensor(0.0, dtype=torch.float).to(self.device)
        criterion_GAN = GANLoss().to(self.device)
        batch, modality0_indices, modality1_indices, paired_in_modality0, paired_in_modality1 = (
            batch
        )
        modality0_volumes = batch["modality0"]
        modality1_volumes = batch["modality1"]
        paired_indices = [index for index in modality0_indices if index in modality1_indices]
        single_modality1_indices = [
            index for index in modality1_indices if index not in paired_indices
        ]
        single_modality0_indices = [
            index for index in modality0_indices if index not in paired_indices
        ]
        if len(single_modality0_indices) > 0:
            single_modality0_volumes = modality0_volumes[single_modality0_indices]
            real_single_modality0 = single_modality0_volumes.to(self.device)
            # real_modality0 = self.post_organize(modality0_volumes).to(self.device)
            fake_single_modality1 = self.gen_021(real_single_modality0)
            loss_fake_1 = criterion_GAN(self.dis_1_cycle(fake_single_modality1), False)
            loss_real_0 = criterion_GAN(self.dis_0_cycle(real_single_modality0), True)
        # if this batch has modality 1
        if len(single_modality1_indices) > 0:
            single_modality1_volumes = modality1_volumes[single_modality1_indices]
            real_single_modality1 = single_modality1_volumes.to(self.device)
            # real_modality1 = self.post_organize(modality1_volumes).to(self.device)
            fake_single_modality0 = self.gen_120(real_single_modality1)
            loss_fake_0 = criterion_GAN(self.dis_0_cycle(fake_single_modality0), False)
            loss_real_1 = criterion_GAN(self.dis_1_cycle(real_single_modality1), True)
        if len(paired_indices) > 0:
            real_paired_modality0 = modality0_volumes[paired_indices].to(self.device)
            real_paired_modality1 = modality1_volumes[paired_indices].to(self.device)
            fake_paired_modality0 = self.gen_120(real_paired_modality1)
            fake_paired_modality1 = self.gen_021(real_paired_modality0)
            loss_paired += criterion_GAN(
                self.dis_0_p2p(real_paired_modality1, fake_paired_modality0), False
            )
            loss_paired += criterion_GAN(
                self.dis_1_p2p(real_paired_modality0, fake_paired_modality1), False
            )
            loss_paired += criterion_GAN(
                self.dis_0_p2p(real_paired_modality1, real_paired_modality0), True
            )
            loss_paired += criterion_GAN(
                self.dis_1_p2p(real_paired_modality0, real_paired_modality1), True
            )
        loss_discrimination = loss_fake_0 + loss_fake_1 + loss_real_0 + loss_real_1 + loss_paired
        return loss_discrimination
        # return super().dis_loss(batch)
