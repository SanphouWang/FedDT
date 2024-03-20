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
    def __init__(self, args, client_id=None, transform=None):
        self.args = args
        self.client_id = client_id
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    # transforms.ToPILImage(),
                    torch.from_numpy,
                    transforms.Resize((256, 256)),
                ]
            )
        )

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
        else:
            self.patient_partition: Dict[str, List] = self.patient_partition["test"]
        self.patient_modality_list: List[Union[int, str]] = (
            []
        )  # [(patient_idx, this patient's modality), ...]
        for modality in self.patient_partition["modality"]:
            self.patient_modality_list = self.patient_modality_list + [
                (patient_idx, modality) for patient_idx in self.patient_partition[modality]
            ]

    def load_brain_volume(self, patient_idx: str, modality: str) -> np.ndarray:
        patient_path = self.patient_path_dict[patient_idx]
        modality_path = os.path.join(patient_path, f"{patient_idx}_{modality}.nii")
        brain_volume = sitk.ReadImage(modality_path)
        brain_volume: np.ndarray = sitk.GetArrayFromImage(brain_volume)
        return brain_volume

    def __len__(self):
        return len(self.patient_modality_list)

    def __getitem__(self, idx):
        patient_idx = self.patient_modality_list[idx][0]
        modality = self.patient_modality_list[idx][1]
        brain_volume = self.load_brain_volume(patient_idx, modality)
        # select slice according to args.slice_idx_begin and args.slice_idx_end
        brain_slice_array = brain_volume[
            self.args.slice_idx_begin : self.args.slice_idx_end, :, :
        ].astype(np.float32)
        # brain_slice_tensor = torch.from_numpy(brain_slice_array)
        # apply transform
        brain_slice = self.transform(brain_slice_array)
        # generate label HGG or LGG
        patient_path = self.patient_path_dict[patient_idx]
        label = patient_path.split("/")[-2]
        label = 0 if label == "HGG" else 1
        return (
            brain_slice,
            modality,
            label,
        )


DATASETS = {"brats2019": Brats2019}

if __name__ == "__main__":
    pass
