import json
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from typing import List, Dict, Union
import SimpleITK as sitk
import numpy as np
import pickle


class Brats2019(Dataset):
    def __init__(self, args, client_id=None, transform=None, test_valid=None):
        self.args = args
        self.client_id = client_id
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    # transforms.ToPILImage(),
                    # transforms.ToPILImage(),
                    # ToTensor(),
                    torch.from_numpy,
                    transforms.Resize((256, 256)),
                    self.normalize,
                ]
            )
        )
        self.test_valid = test_valid
        with open(
            os.path.join(args.data_path, "preprocessed_files", "patient_path.pkl"), "rb"
        ) as f:
            self.patient_path_dict = pickle.load(f)
        with open(
            os.path.join(args.data_path, "preprocessed_files", "patient_partition.pkl"), "rb"
        ) as f:
            self.patient_partition = pickle.load(f)
        # if client_id is not None, load training set
        if self.client_id is not None:
            self.patient_partition: Dict[str, List] = self.patient_partition[client_id]
            self.patient_modality_list: List[Union[int, str]] = (
                []
            )  # [(patient_idx, this patient's modality), ...]
            for modality in self.patient_partition["modality"]:
                self.patient_modality_list = self.patient_modality_list + [
                    (patient_idx, modality) for patient_idx in self.patient_partition[modality]
                ]
        else:  # load test set
            if self.test_valid is None:
                raise ValueError("test_valid should be specified when client_id is None")
            if self.test_valid == "test":
                self.patient_partition: Dict[str, List] = self.patient_partition["test"]
            else:
                self.patient_partition: Dict[str, List] = self.patient_partition["valid"]

    def load_brain_volume(self, patient_idx: str, modality: str) -> np.ndarray:
        patient_path = self.patient_path_dict[patient_idx]
        modality_path = os.path.join(patient_path, f"{patient_idx}_{modality}.nii")
        brain_volume = sitk.ReadImage(modality_path)
        brain_volume = sitk.Cast(brain_volume, sitk.sitkFloat32)
        brain_volume: np.ndarray = sitk.GetArrayFromImage(brain_volume)
        return brain_volume

    def __len__(self):
        if self.client_id is not None:
            return len(self.patient_modality_list)
        else:
            return len(self.patient_partition["patient"])

    def normalize(self, brain_slice):
        max_value, _ = torch.max(brain_slice.view(brain_slice.shape[0], -1), dim=1, keepdim=True)
        max_value = max_value.view(brain_slice.shape[0], 1, 1)
        normalized = brain_slice / max_value
        return normalized

    def __getitem__(self, idx):

        if self.client_id is not None:
            """
            Training Set
            """
            patient_idx = self.patient_modality_list[idx][0]
            modality = self.patient_modality_list[idx][1]
            brain_volume = self.load_brain_volume(patient_idx, modality)
            # select slice according to args.slice_idx_begin and args.slice_idx_end
            brain_slice_array = brain_volume[
                self.args.slice_idx_begin : self.args.slice_idx_end, :, :
            ].astype(np.float32)
            # brain_slice_tensor = torch.from_numpy(brain_slice_array)
            # apply transform
            # brain_slice_array = brain_slice_array / brain_slice_array.mean(axis=(1, 2))
            brain_slice = self.transform(brain_slice_array)
            # generate label HGG or LGG
            patient_path = self.patient_path_dict[patient_idx]
            label = patient_path.split("/")[-2]
            label = 0 if label == "HGG" else 1
            return brain_slice, modality, label

        else:
            """
            Test Set or Validation Set
            """
            patient_idx = self.patient_partition["patient"][idx]
            modality0 = self.patient_partition["modality"][0]
            modality1 = self.patient_partition["modality"][1]
            brain_volume0 = self.load_brain_volume(patient_idx, modality0)
            brain_slice_array0 = brain_volume0[
                self.args.slice_idx_begin : self.args.slice_idx_end, :, :
            ].astype(np.float32)
            brain_slice0 = self.transform(brain_slice_array0)
            brain_volume1 = self.load_brain_volume(patient_idx, modality1)
            brain_slice_array1 = brain_volume1[
                self.args.slice_idx_begin : self.args.slice_idx_end, :, :
            ].astype(np.float32)
            brain_slice1 = self.transform(brain_slice_array1)
            patient_path = self.patient_path_dict[patient_idx]
            label = patient_path.split("/")[-2]
            label = 0 if label == "HGG" else 1
            return {
                "modality0": brain_slice0,
                "modality1": brain_slice1,
                "modality": [modality0, modality1],
                "label": label,
            }


DATASETS = {"brats2019": Brats2019}

if __name__ == "__main__":
    pass
