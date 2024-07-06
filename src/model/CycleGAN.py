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
)
from src.model.big_unet_generator import BigUNetGenerator
from src.utils.tools import move2cpu, move2device
from src.utils.metrics import METRIC_MAPPING

DATASET_MODEL_MAP = {
    "brats2019": {
        "generator": ResnetGenerator(n_blocks=9),
        "discriminator": NLayerDiscriminator(n_layers=6),
    },
    "adni_roi": {
        "generator": MLPGenerator,
        "discriminator": MLPDiscriminator,
    },
}


class CycleGAN:
    def __init__(
        self,
        args,
        logger,
        device=None,
        mode="train",
    ) -> None:
        self.args = args
        self.device = device
        self.mode = mode
        self.logger = logger
        self.gen_021 = DATASET_MODEL_MAP[self.args.dataset]["generator"]()
        self.gen_120 = DATASET_MODEL_MAP[self.args.dataset]["generator"]()
        self.generator_list = [self.gen_120, self.gen_021]
        self.generate_image_flag = False  # for fix batch for generating image
        self.current_round = 0  # record current step for evaluating training process
        if mode == "train":
            self.initialize_training_component()

    def initialize_training_component(self):
        self.dis_0 = DATASET_MODEL_MAP[self.args.dataset]["discriminator"]()
        self.dis_1 = DATASET_MODEL_MAP[self.args.dataset]["discriminator"]()
        self.dis_list = [self.dis_0, self.dis_1]
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
        self.optimizer_dis_0 = torch.optim.Adam(
            self.dis_0.parameters(),
            lr=self.args.dis_lr,
            # momentum=0.9,
            weight_decay=0.00001,
        )
        self.optimizer_dis_1 = torch.optim.Adam(
            self.dis_1.parameters(),
            lr=self.args.dis_lr,
            # momentum=0.9,
            weight_decay=0.00001,
        )
        self.generator_optimizer_list = [self.optimizer_gen_021, self.optimizer_gen_120]
        self.dis_optimizer_list = [self.optimizer_dis_0, self.optimizer_dis_1]

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - self.args.gen_samelr_round) / float(
                self.args.gen_round - self.args.gen_samelr_round + 1
            )
            return lr_l

        self.shcheduler = [
            LambdaLR(self.optimizer_gen_021, lr_lambda=lambda_rule),
            LambdaLR(self.optimizer_gen_120, lr_lambda=lambda_rule),
            LambdaLR(self.optimizer_dis_0, lr_lambda=lambda_rule),
            LambdaLR(self.optimizer_dis_1, lr_lambda=lambda_rule),
        ]

    def eval_generator(self, dataloader):
        for generator in self.generator_list:
            generator.eval()
        # self.gen_021.eval()
        # self.gen_120.eval()
        for dis in self.dis_list:
            dis.eval()
        # self.dis_0.eval()
        # self.dis_1.eval()
        loss_generation_cache = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                loss_generation: Dict = self.gen_loss(batch, evaluate_training=True)
                if batch_idx == 0:
                    for key in loss_generation.keys():
                        loss_generation_cache[key] = loss_generation[key]
                else:
                    for key in loss_generation.keys():
                        loss_generation_cache[key] += loss_generation[key]
        for key in loss_generation_cache.keys():
            loss_generation_cache[key] = loss_generation_cache[key] / len(dataloader)
        return loss_generation_cache

    def eval_discriminator(self, dataloader):
        for generator in self.generator_list:
            generator.eval()
        for dis in self.dis_list:
            dis.eval()
        # self.gen_021.eval()
        # self.gen_120.eval()
        # self.dis_0.eval()
        # self.dis_1.eval()
        loss_discrimination_cache = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                loss_discrimination = self.dis_loss(batch)
                loss_discrimination_cache += loss_discrimination
        return {"discriminator_loss": loss_discrimination_cache / len(dataloader)}

    def update_learning_rate(self):
        for scheduler in self.shcheduler:
            old_lr = scheduler.get_last_lr()
            scheduler.step()
            new_lr = scheduler.get_last_lr()
            if old_lr != new_lr:
                self.logger.log("Update learning rate to " + str(scheduler.get_last_lr()))

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
        for generator in self.generator_list:
            generator.train()
        for dis in self.dis_list:
            dis.eval()

        # self.gen_021.train()
        # self.gen_120.train()
        # self.dis_0.eval()
        # self.dis_1.eval()
        for batch_idx, batch in enumerate(dataloader):
            loss_generation = self.gen_loss(batch)
            loss_generation_cache = loss_generation_cache + loss_generation.item()
            for optimizer in self.generator_optimizer_list:
                optimizer.zero_grad()
            # self.optimizer_gen_021.zero_grad()
            # self.optimizer_gen_120.zero_grad()
            loss_generation.backward()
            for optimizer in self.generator_optimizer_list:
                optimizer.step()
            # self.optimizer_gen_021.step()
            # self.optimizer_gen_120.step()

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
        for generator in self.generator_list:
            generator.eval()
        for dis in self.dis_list:
            dis.train()
        # self.gen_021.eval()
        # self.gen_120.eval()
        # self.dis_0.train()
        # self.dis_1.train()
        if (self.current_round + 1) % self.args.dis_train_gap == 0:
            for batch_idx, batch in enumerate(dataloader):
                loss_discrimination = self.dis_loss(batch)
                loss_discrimination_cache += loss_discrimination.item()
                for optimizer in self.dis_optimizer_list:
                    optimizer.zero_grad()
                # self.optimizer_dis_0.zero_grad()
                # self.optimizer_dis_1.zero_grad()
                loss_discrimination.backward()
                for optimizer in self.dis_optimizer_list:
                    optimizer.step()
                # self.optimizer_dis_0.step()
                # self.optimizer_dis_1.step()
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
        criterion_GAN = GANLoss().to(self.device)
        criterion_cycle = torch.nn.L1Loss().to(self.device)
        criterion_identity = torch.nn.L1Loss().to(self.device)
        criterion_paired = torch.nn.L1Loss().to(self.device)
        batch, modality0_indices, modality1_indices, paired_in_modality0, paired_in_modality1 = (
            batch
        )
        modality0_volumes = batch["modality0"]
        modality1_volumes = batch["modality1"]

        # initialize loss term
        loss_GAN_021 = torch.tensor(0.0, dtype=torch.float)
        loss_id_021 = torch.tensor(0.0, dtype=torch.float)
        loss_cycle_021 = torch.tensor(0.0, dtype=torch.float)
        loss_GAN_120 = torch.tensor(0.0, dtype=torch.float)
        loss_id_120 = torch.tensor(0.0, dtype=torch.float)
        loss_cycle_120 = torch.tensor(0.0, dtype=torch.float)

        if len(modality0_indices) > 0:
            modality0_volumes = modality0_volumes[modality0_indices]
            real_modality0 = modality0_volumes.to(self.device)
            # real_modality0 = self.post_organize(modality0_volumes).to(self.device)
            fake_modality1 = self.gen_021(real_modality0)
            loss_GAN_021 = criterion_GAN(self.dis_1(fake_modality1), True)
            # loss_id_021 = criterion_identity(self.gen_120(real_modality0), real_modality0)
            loss_cycle_021 = criterion_cycle(self.gen_120(fake_modality1), real_modality0)
        # if this batch has modality 1
        if len(modality1_indices) > 0:
            modality1_volumes = modality1_volumes[modality1_indices]
            real_modality1 = modality1_volumes.to(self.device)
            # real_modality1 = self.post_organize(modality1_volumes).to(self.device)
            fake_modality0 = self.gen_120(real_modality1)
            loss_GAN_120 = criterion_GAN(self.dis_0(fake_modality0), True)
            # loss_id_120 = criterion_identity(self.gen_021(real_modality1), real_modality1)
            loss_cycle_120 = criterion_cycle(self.gen_021(fake_modality0), real_modality1)
        loss_generation = (
            self.args.lambda_gan * (loss_GAN_021 + loss_GAN_120)
            # + self.args.lambda_identity * (loss_id_021 + loss_id_120)
            + self.args.lambda_cycle * (loss_cycle_021 + loss_cycle_120)
        )
        return_dict = {
            "gan_loss": self.args.lambda_gan * (loss_GAN_021 + loss_GAN_120),
            # "identity_loss": self.args.lambda_identity * (loss_id_021 + loss_id_120),
            "cycle_loss": self.args.lambda_cycle * (loss_cycle_021 + loss_cycle_120),
        }
        # if (
        #     len(paired_in_modality0) > 0
        #     and len(paired_in_modality1) > 0
        #     and self.args.lambda_paired > 0
        # ):
        #     real_paired_modality0 = real_modality0[paired_in_modality0]
        #     real_paired_modality1 = real_modality1[paired_in_modality1]
        #     fake_paired_modality0 = fake_modality0[paired_in_modality0]
        #     fake_paired_modality1 = fake_modality1[paired_in_modality1]
        #     loss_paired_0 = criterion_paired(fake_paired_modality0, real_paired_modality0)
        #     loss_paired_1 = criterion_paired(fake_paired_modality1, real_paired_modality1)
        #     loss_generation += self.args.lambda_paired * (loss_paired_0 + loss_paired_1)
        #     return_dict["paired_loss"] = self.args.lambda_paired * (loss_paired_0 + loss_paired_1)

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
        batch, modality0_indices, modality1_indices, paired_in_modality0, paired_in_modality1 = (
            batch
        )

        modality0_volumes = batch["modality0"]
        modality1_volumes = batch["modality1"]
        if len(modality0_indices) > 0:
            modality0_volumes = modality0_volumes[modality0_indices]
            real_modality0 = modality0_volumes.to(self.device)
            # real_modality0 = self.post_organize(modality0_volumes).to(self.device)
            fake_modality1 = self.gen_021(real_modality0)
            loss_fake_1 = criterion_GAN(self.dis_1(fake_modality1), False)
            loss_real_0 = criterion_GAN(self.dis_0(real_modality0), True)
        # if this batch has modality 1
        if len(modality1_indices) > 0:
            modality1_volumes = modality1_volumes[modality1_indices]
            real_modality1 = modality1_volumes.to(self.device)
            # real_modality1 = self.post_organize(modality1_volumes).to(self.device)
            fake_modality0 = self.gen_120(real_modality1)
            loss_fake_0 = criterion_GAN(self.dis_0(fake_modality0), False)
            loss_real_1 = criterion_GAN(self.dis_1(real_modality1), True)
        loss_discrimination = 0.5 * (loss_fake_0 + loss_fake_1 + loss_real_0 + loss_real_1)
        return loss_discrimination

    def move2cpu(self):
        self.gen_021 = move2cpu(self.gen_021)
        self.gen_120 = move2cpu(self.gen_120)
        if self.mode == "train":
            self.dis_0 = move2cpu(self.dis_0)
            self.dis_1 = move2cpu(self.dis_1)

    def move2device(self):
        if self.device is None:
            raise ValueError("Device is not set")
        self.gen_021 = move2device(self.device, self.args.multi_gpu, self.gen_021)
        self.gen_120 = move2device(self.device, self.args.multi_gpu, self.gen_120)
        if self.mode == "train":
            self.dis_0 = move2device(self.device, self.args.multi_gpu, self.dis_0)
            self.dis_1 = move2device(self.device, self.args.multi_gpu, self.dis_1)

    def get_weights(self) -> List[OrderedDict]:
        generator_weight_list = [generator.state_dict() for generator in self.generator_list]
        return generator_weight_list + [dis.state_dict() for dis in self.dis_list]

        # return [
        #     self.gen_120.state_dict(),
        #     self.gen_021.state_dict(),
        #     self.dis_0.state_dict(),
        #     self.dis_1.state_dict(),
        # ]

    def download_weights(self, weights: List[OrderedDict], resume=False):
        generator_num = len(self.generator_list)
        dis_num = len(self.dis_list)
        for index, weight in enumerate(weights):
            if index < generator_num:
                try:
                    self.generator_list[index].load_state_dict(weight)
                except:
                    raise Warning("Fail to resume generator weights. Will use default weights")
                    pass
            else:
                if (self.args.upload_dis and self.mode == "train") or resume:
                    try:
                        self.dis_list[index - generator_num].load_state_dict(weight)
                    except:
                        raise Warning(
                            "Fail to resume discriminator weights. Will use default weights"
                        )
                        pass

    def get_generator(self):
        return self.gen_021, self.gen_120

    def test(self, dataloader):
        self.gen_021.eval()
        self.gen_120.eval()
        metric_func = METRIC_MAPPING[self.args.dataset]
        metric_cache = {}
        for key in metric_func.keys():
            metric_func[key] = metric_func[key]
            metric_cache[key] = {"021": [], "120": []}

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                real_modality0 = batch["modality0"]
                real_modality1 = batch["modality1"]
                real_modality0 = dataloader.dataset.post_organize(real_modality0).to(self.device)
                real_modality1 = dataloader.dataset.post_organize(real_modality1).to(self.device)

                fake_modality1 = self.gen_021(real_modality0)
                fake_modality0 = self.gen_120(real_modality1)
                real_modality0 = real_modality0.cpu()
                real_modality1 = real_modality1.cpu()
                fake_modality1 = fake_modality1.cpu()
                fake_modality0 = fake_modality0.cpu()
                for key in metric_func.keys():
                    metric_cache[key]["021"].append(
                        real_modality0.shape[0]
                        / len(dataloader.dataset)
                        * metric_func[key](fake_modality1, real_modality1)
                    )
                    metric_cache[key]["120"].append(
                        real_modality0.shape[0]
                        / len(dataloader.dataset)
                        * metric_func[key](fake_modality0, real_modality0)
                    )
        for key in metric_func.keys():
            metric_cache[key]["021"] = torch.sum(torch.stack(metric_cache[key]["021"]), dim=0).cpu()
            metric_cache[key]["120"] = torch.sum(torch.stack(metric_cache[key]["120"]), dim=0).cpu()
        new_dict = {}
        for key in metric_cache.keys():
            new_dict[f"{key}_021"] = round(metric_cache[key]["021"].item(), 4)
            new_dict[f"{key}_120"] = round(metric_cache[key]["120"].item(), 4)
        return new_dict

    def plot_test_result(self, test_result_list, test_rounds, out_dir):
        cat_metric = {}
        test_rounds = [int(round) for round in test_rounds]
        metric_func = METRIC_MAPPING[self.args.dataset]

        for metric_name in test_result_list[0].keys():
            cat_metric[metric_name] = []
        for metric_at_round in test_result_list:
            for key, value in metric_at_round.items():
                cat_metric[key].append(value)
        for metric_name in metric_func.keys():
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
            plt.xticks(test_rounds)
            plt.xlabel("Test Round")
            # plt.ylabel(metric_name)
            plt.legend()
            plt.title(metric_name.upper())
            plt.savefig(f"{out_dir}/{metric_name}.png")

    def generate_image(self, out_dir, dataloader):

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if self.generate_image_flag == False:
            self.generate_image_flag = True
            self.batch4generating_image = next(iter(dataloader))
            self.slice_idx4generating_image = random.sample(
                range((self.args.slice_idx_end - self.args.slice_idx_begin)),
                4,
            )
        self.gen_021.eval()
        self.gen_120.eval()
        with torch.no_grad():
            batch = self.batch4generating_image
            real_modality0 = batch["modality0"]
            real_modality1 = batch["modality1"]
            real_modality0 = real_modality0.view(
                -1, 1, real_modality0.shape[2], real_modality0.shape[3]
            ).to(self.device)
            real_modality1 = real_modality1.view(
                -1, 1, real_modality1.shape[2], real_modality1.shape[3]
            ).to(self.device)
            fake_modality1 = self.gen_021(real_modality0)
            fake_modality0 = self.gen_120(real_modality1)
            # randomly select 4 slices
            real_modality0 = real_modality0[self.slice_idx4generating_image]
            real_modality1 = real_modality1[self.slice_idx4generating_image]
            fake_modality1 = fake_modality1[self.slice_idx4generating_image]
            fake_modality0 = fake_modality0[self.slice_idx4generating_image]
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
