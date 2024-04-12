import json
import functools
from collections import OrderedDict
from typing import List, Optional
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
import torch.distributions as distributions
from torch import sigmoid

# from tools import PROJECT_DIR
import torch

INPUT_CHANNELS = {"brats2019": 1}

NUM_CLASSES = {"brats2019": 2}


class DecoupledModel(nn.Module):
    def __init__(self):
        super(DecoupledModel, self).__init__()
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.dropout: List[nn.Module] = []

    def need_all_features(self):
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def get_feature_hook_fn(model, input, output):
            if self.need_all_features_flag:
                self.all_features.append(output.clone().detach())

        for module in target_modules:
            module.register_forward_hook(get_feature_hook_fn)

    def check_avaliability(self):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module
            for module in list(self.base.modules()) + list(self.classifier.modules())
            if isinstance(module, nn.Dropout)
        ]

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(self.classifier(self.base(x)))
        # return self.classifier(self.base(x))

    def get_final_features(self, x: Tensor, detach=True) -> Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        out = self.base(x)

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: Tensor) -> Optional[List[Tensor]]:
        feature_list = None
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        self.need_all_features_flag = True
        _ = self.base(x)
        self.need_all_features_flag = False

        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return feature_list


class ResNet50(DecoupledModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        dataset_name = self.args.dataset
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # resnet = models.resnet50()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base = resnet
        self.classifier = nn.Linear(self.base.fc.in_features, NUM_CLASSES[dataset_name])

        self.base.fc = nn.Identity()


class MobileNet(DecoupledModel):
    def __init__(self, args):
        super().__init__()
        # pretrained = True
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.base = mobilenet
        self.classifier = nn.Linear(mobilenet.classifier[-1].in_features, NUM_CLASSES[args.dataset])
        self.base.classifier[-1] = nn.Identity()


class modifybasicstem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""

    def __init__(self):
        super(modifybasicstem, self).__init__(
            nn.Conv3d(
                1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


def resnet_mixed_conv():
    model = models.video.mc3_18(weights=models.video.MC3_18_Weights)
    model.stem = modifybasicstem()
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 2))
    return model


# class ResNetMixedConv(DecoupledModel):
#     def __init__(self, args):
#         super().__init__()
#         model = models.video.mc3_18(weights=models.ResNet50_Weights.DEFAULT)
#         model.stem = modifybasicstem()
#         self.classifier = model


if __name__ == "__main__":

    class Args:
        dataset = "brats2019"

    args = Args
    # model = ResNet50(args)
    model = resnet_mixed_conv()
    # Generate random tensor
    input_tensor = torch.randn(3, 1, 20, 256, 256)

    # Pass the input tensor through the model
    output_tensor = model(input_tensor)
    print(output_tensor)
    # Print the shape of the output tensor
    print(output_tensor.shape)
