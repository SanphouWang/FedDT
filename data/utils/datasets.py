import json
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from typing import List, Dict, Union
import SimpleITK as sitk
import numpy as np
import pickle
import pandas as pd


class Brats2019(Dataset):
    def __init__(self, args, client_id=None, transform=None):
        self.args = args
        self.client_id = client_id
        self.size = 256
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    torch.from_numpy,
                    transforms.Resize((self.size, self.size)),
                    self.normalize,
                ]
            )
        )

        self.image_shape = (1, self.size, self.size)
        with open(
            os.path.join(args.data_path, args.preprocessed_file_directory, "patient_path.pkl"), "rb"
        ) as f:
            self.patient_path_dict = pickle.load(f)
        with open(
            os.path.join(args.data_path, args.preprocessed_file_directory, "patient_partition.pkl"),
            "rb",
        ) as f:
            self.patient_partition = pickle.load(f)

        # judge the type of dataset
        if self.client_id != "test" and self.client_id != "valid":
            self.mode = "train"
            self.transform.transforms.insert(2, transforms.RandomHorizontalFlip(p=0.5))
        elif self.client_id == "test":
            self.mode = "test"
        elif self.client_id == "valid":
            self.mode = "valid"
        else:
            raise ValueError("client_id should be either 'test' or 'valid' or a client id")
        self.num_classes = 2
        # read dictinaries
        if self.mode == "train":
            self.patient_partition: Dict[str, List] = self.patient_partition[client_id]
            self.patient_modality0_list = self.patient_partition[
                self.args.modality0
            ]  # patient with modality0
            self.patient_modality1_list = self.patient_partition[
                self.args.modality1
            ]  # patient with modality1
            self.patient_list = list(
                set(self.patient_modality0_list + self.patient_modality1_list)
            )  # union of two list
            self.complet_subject = self.patient_partition["complet_subject"]
            self.modality_list = self.patient_partition["modality"]
        else:
            if self.mode == "test":
                self.patient_partition: Dict[str, List] = self.patient_partition["test"]
            else:
                self.patient_partition: Dict[str, List] = self.patient_partition["valid"]
            self.patient_list = self.patient_partition["patient"]

    def get_slice(self, idx):
        idx = idx + 1
        slices_num = self.args.slice_idx_end - self.args.slice_idx_begin
        patient_idx = idx // slices_num
        patient_name = self.patient_list[patient_idx]
        patient_path = self.patient_path_dict[patient_name]
        label = patient_path.split("/")[-2]
        label = torch.tensor([0]) if label == "HGG" else torch.tensor([1])
        slice_idx = idx % slices_num
        if slice_idx == 0:
            slice_idx = slices_num
        slice_idx = slice_idx + self.args.slice_idx_begin - 1

    def load_brain_volume(self, patient_idx: str, modality: str) -> np.ndarray:
        patient_path = self.patient_path_dict[patient_idx]
        modality_path = os.path.join(patient_path, f"{patient_idx}_{modality}.nii")
        brain_volume = sitk.ReadImage(modality_path)
        brain_volume = sitk.Cast(brain_volume, sitk.sitkFloat32)
        brain_volume: np.ndarray = sitk.GetArrayFromImage(brain_volume)
        return brain_volume

    def load_data(self, patient_name: str, modality: str):
        brain_volume = self.load_brain_volume(patient_name, modality)
        brain_slice_array = brain_volume[
            self.args.slice_idx_begin : self.args.slice_idx_end, :, :
        ].astype(np.float32)
        brain_slice = self.transform(brain_slice_array)
        return brain_slice

    def __len__(self):
        return len(self.patient_list)

    def get_len(self):
        if self.mode != "train":
            raise ValueError("get_len() is only for training set")
        return len(self.patient_modality0_list) + len(self.patient_modality1_list)

    def normalize(self, brain_slice):
        max_value, _ = torch.max(brain_slice.view(brain_slice.shape[0], -1), dim=1, keepdim=True)
        max_value = max_value.view(brain_slice.shape[0], 1, 1)
        normalized = brain_slice / max_value
        return normalized

    @staticmethod
    def generate_indices(args, batch):
        """
        Parameters:
        -batch (Dict): contain following keys:
            - "modality0" (torch.Tensor): The brain slice for modality 0.
            - "modality1" (torch.Tensor): The brain slice for modality 1.
            - "modality" (list): A list containing the names of the modalities for this patient.
            - "label" (int): The label for the item (0 for "HGG" and 1 for other labels).
            - "paired_flag" (int): A flag indicating whether the patient has both modality in this client. 1->has both, 0-> has single.
                of course it should be 1 for test set and validation set.
        Returns:
        - modality0_indices (List): indices of modality0 volume in the batch
        - modality1_indices (List): indices of modality1 volume in the batch
        - paired_in_modality0 (List): indices of volumes in modality0_indices that have both modalites.
            i.e. for any i in range(len(paired_in_modality0)), we have modality0_indices[paired_in_modality0[i]] this volume has both modalities
        - paired_in_modality1 (List): similar to paired_in_modality0
        """
        modality_list = batch["modality"]
        # paired_flag = batch["paired_flag"]
        modality0_indices = [
            i
            for i in range(len(modality_list))
            if modality_list[i] == args.modality0 or modality_list[i] == "both"
        ]
        modality1_indices = [
            i
            for i in range(len(modality_list))
            if modality_list[i] == args.modality1 or modality_list[i] == "both"
        ]
        paired_indices = [i for i in range(len(modality_list)) if modality_list[i] == "both"]
        # find the index of each element in paired_indices in modality0_indices and modality1_indices
        paired_in_modality0 = [modality0_indices.index(i) for i in paired_indices]
        paired_in_modality1 = [modality1_indices.index(i) for i in paired_indices]
        return modality0_indices, modality1_indices, paired_in_modality0, paired_in_modality1

    @classmethod
    def class_organize_batch(
        cls,
        batch,
        args,
        device,
        generator_021=None,
        generator_120=None,
    ):
        """
        Organize the batch for classification training.
        """
        modality0_indices, modality1_indices, _, _ = cls.generate_indices(args, batch)
        paired_indices = list(set(modality0_indices) & set(modality1_indices))
        single_modality0_indices = [i for i in modality0_indices if i not in paired_indices]
        single_modality1_indices = [i for i in modality1_indices if i not in paired_indices]
        modality0_volumes_original = batch["modality0"]
        modality1_volumes_original = batch["modality1"]
        labels_original = batch["label"]
        modality0_volumes = modality0_volumes_original[modality0_indices]
        modality0_volumes = modality0_volumes.unsqueeze(2)
        real_modality0 = modality0_volumes.permute(0, 2, 1, 3, 4)
        modality0_labels = labels_original[modality0_indices].squeeze(1)
        modality1_volumes = modality1_volumes_original[modality1_indices]
        modality1_volumes = modality1_volumes.unsqueeze(2)
        real_modality1 = modality1_volumes.permute(0, 2, 1, 3, 4)
        modality1_labels = labels_original[modality1_indices].squeeze(1)
        image = torch.cat([real_modality0, real_modality1], dim=0)
        label = torch.cat([modality0_labels, modality1_labels], dim=0)
        if args.use_generator and generator_021 is not None and generator_120 is not None:
            with torch.no_grad():
                # generate fake modality1 and its labels
                single_modality0_volumes = modality0_volumes_original[single_modality0_indices]
                single_modality0_volumes = single_modality0_volumes.view(
                    -1, 1, single_modality0_volumes.shape[2], single_modality0_volumes.shape[3]
                ).to(device)
                fake_modality1 = generator_021(single_modality0_volumes)
                single_modality0_labels = labels_original[single_modality0_indices]
                single_modality0_labels_repeat = single_modality0_labels.repeat(
                    1, args.slice_idx_end - args.slice_idx_begin
                )
                single_modality0_labels_vectorized = single_modality0_labels_repeat.view(-1, 1).to(
                    device
                )
                single_modality1_volumes = modality1_volumes_original[single_modality1_indices]
                single_modality1_volumes = single_modality1_volumes.view(
                    -1, 1, single_modality1_volumes.shape[2], single_modality1_volumes.shape[3]
                ).to(device)
                fake_modality0 = generator_120(single_modality1_volumes)
                single_modality1_labels = labels_original[single_modality1_indices]
                single_modality1_labels_repeat = single_modality1_labels.repeat(
                    1, args.slice_idx_end - args.slice_idx_begin
                )
                single_modality1_labels_vectorized = single_modality1_labels_repeat.view(-1, 1)
                image = torch.cat([image, fake_modality0, fake_modality1], dim=0).to(device)
                label = torch.cat(
                    [
                        label,
                        single_modality1_labels_vectorized,
                        single_modality0_labels_vectorized,
                    ],
                    dim=0,
                ).to(device)
        return image, label

    @staticmethod
    def post_organize(self, volumes):
        return volumes.view(-1, 1, volumes.shape[2], volumes.shape[3])

    def __getitem__(self, idx):
        """
        Get the item at the given index.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        - dict: A dictionary containing the following keys:
            - "modality0" (torch.Tensor): The brain slice for modality 0.
            - "modality1" (torch.Tensor): The brain slice for modality 1.
            - "modality" (list): A list containing the names of the modalities for this patient. shoul be  self.args.modality0 or self.args.modality1 or "both"
            - "label" (int): The label for the item (0 for "HGG" and 1 for other labels).
        """
        patient_name = self.patient_list[idx]
        patient_path = self.patient_path_dict[patient_name]
        label = patient_path.split("/")[-2]
        label = torch.tensor([0]) if label == "HGG" else torch.tensor([1])
        if self.mode == "train":
            shape = (self.args.slice_idx_end - self.args.slice_idx_begin, self.size, self.size)

            if patient_name in self.patient_modality0_list:

                brain_slice0 = self.load_data(patient_name, self.args.modality0)
                paired_flag = 1
                modality = self.args.modality0
            else:
                brain_slice0 = torch.zeros(shape)
                paired_flag = 0
            if patient_name in self.patient_modality1_list:
                brain_slice1 = self.load_data(patient_name, self.args.modality1)
                paired_flag = 1 * paired_flag
                modality = "both" if paired_flag else self.args.modality1
            else:
                brain_slice1 = torch.zeros(shape)
        else:
            modality0 = self.patient_partition["modality"][0]
            modality1 = self.patient_partition["modality"][1]
            brain_slice0 = self.load_data(patient_name, modality0)
            brain_slice1 = self.load_data(patient_name, modality1)
            modality = "both"
        return {
            "modality0": brain_slice0,
            "modality1": brain_slice1,
            "modality": modality,
            "label": label,
        }


# def brats2019_collate_fn(batch):
#     """
#     Collate function for the Brats2019 dataset.

#     Parameters:
#     - batch (List[Dict]): A list of dictionaries containing the following
#     """
#     modality0 = torch.stack([item["modality0"] for item in batch])
#     modality1 = torch.stack([item["modality1"] for item in batch])
#     modality = [item["modality"] for item in batch]
#     label = torch.stack([item["label"] for item in batch])

#     return {"modality0": modality0, "modality1": modality1, "modality": modality, "label": label}


class ADNI_ROI(Dataset):
    def __init__(
        self,
        args,
        client_id=None,
        transform=None,
    ):
        super(ADNI_ROI, self).__init__()
        self.args = args
        self.client_id = client_id
        self.transform = transform if transform else transforms.Compose([])

        with open(
            os.path.join(args.data_path, args.preprocessed_file_directory, "patient_partition.pkl"),
            "rb",
        ) as f:
            self.patient_partition = pickle.load(f)
        self.data = pd.read_csv(os.path.join(args.data_path, "preprocessed.csv"))
        self.Statistic3M = pd.read_csv(os.path.join(args.data_path, "Statistic3M.csv"))
        self.label_mapping = {"MCI": 0, "CN": 1, "AD": 2, "Empty": 3}
        # judge the type of dataset
        if self.client_id != "test" and self.client_id != "valid":
            self.mode = "train"
        elif self.client_id == "test":
            self.mode = "test"
        elif self.client_id == "valid":
            self.mode = "valid"
        else:
            raise ValueError("client_id should be either 'test' or 'valid' or a client id")
        self.num_classes = 3
        # read dictinaries
        if self.mode == "train":
            self.patient_partition: Dict[str, List] = self.patient_partition[client_id]
            self.patient_modality0_list = self.patient_partition[
                self.args.modality0
            ]  # patient with modality0
            self.patient_modality1_list = self.patient_partition[
                self.args.modality1
            ]  # patient with modality1
            self.patient_list = list(
                set(self.patient_modality0_list + self.patient_modality1_list)
            )  # union of two list
            self.complete_patient = self.patient_partition["complete_patient"]
            self.modality_list = self.patient_partition["modality"]
        else:
            if self.mode == "test":
                self.patient_partition: Dict[str, List] = self.patient_partition["test"]
            else:
                self.patient_partition: Dict[str, List] = self.patient_partition["valid"]
            self.patient_list = self.patient_partition["patient"]

    def __len__(self):
        return len(self.patient_list)

    def data_amount(self, task="generation"):
        if task == "generation":
            if self.mode == "train":
                return len(self.patient_modality0_list) + len(self.patient_modality1_list)
            else:
                return len(self.patient_list) * 2
        else:
            # for classification
            count = 0
            for patient_id in self.patient_list:
                label = self.Statistic3M[self.Statistic3M["IID"] == patient_id]["diagnosis"].values[
                    0
                ]
                if label != "Empty":
                    if patient_id in self.patient_modality0_list:
                        count += 1
                    if patient_id in self.patient_modality1_list:
                        count += 1
            return count

    @staticmethod
    def generate_indices(args, batch):
        """
        Parameters:
        -batch (Dict): contain following keys:
            - "modality0" (torch.Tensor): The brain slice for modality 0.
            - "modality1" (torch.Tensor): The brain slice for modality 1.
            - "modality" (list): A list containing the names of the modalities for this patient.
            - "label" (int): The label for the item (0 for "HGG" and 1 for other labels).
        Returns:
        - modality0_indices (List): indices of modality0 volume in the batch
        - modality1_indices (List): indices of modality1 volume in the batch
        - paired_in_modality0 (List): indices of volumes in modality0_indices that have both modalites.
            i.e. for any i in range(len(paired_in_modality0)), we have modality0_indices[paired_in_modality0[i]] this volume has both modalities
        - paired_in_modality1 (List): similar to paired_in_modality0
        """
        modality_list = batch["modality"]
        # paired_flag = batch["paired_flag"]
        modality0_indices = [
            i
            for i in range(len(modality_list))
            if modality_list[i] == args.modality0 or modality_list[i] == "both"
        ]
        modality1_indices = [
            i
            for i in range(len(modality_list))
            if modality_list[i] == args.modality1 or modality_list[i] == "both"
        ]
        paired_indices = [i for i in range(len(modality_list)) if modality_list[i] == "both"]
        # find the index of each element in paired_indices in modality0_indices and modality1_indices
        paired_in_modality0 = [modality0_indices.index(i) for i in paired_indices]
        paired_in_modality1 = [modality1_indices.index(i) for i in paired_indices]
        return modality0_indices, modality1_indices, paired_in_modality0, paired_in_modality1

    @staticmethod
    def post_organize(volumes):
        return volumes

    def __getitem__(self, index):
        patient_id = self.patient_list[index]
        patient_data = self.data[self.data["IID"] == patient_id]
        label = self.Statistic3M[self.Statistic3M["IID"] == patient_id]["diagnosis"].values[0]
        label = self.label_mapping[label]
        label = torch.tensor(label)
        if self.mode == "train":
            modality0 = np.zeros(90)
            modality1 = np.zeros(90)
            complete_flag = 0
            if patient_id in self.patient_modality0_list:
                modality0 = patient_data[patient_data["modality"] == self.args.modality0].values[
                    0, 2:
                ]
                complete_flag = 1
                modality = self.args.modality0
            if patient_id in self.patient_modality1_list:
                modality1 = patient_data[patient_data["modality"] == self.args.modality1].values[
                    0, 2:
                ]
                complete_flag = 1 * complete_flag
                if complete_flag == 1:
                    modality = "both"
                else:
                    modality = self.args.modality1
            modality0 = torch.tensor(modality0.astype(np.float32))
            modality1 = torch.tensor(modality1.astype(np.float32))
        else:
            modality = "both"
            modality0 = patient_data[patient_data["modality"] == self.args.modality0].values[0, 2:]
            modality1 = patient_data[patient_data["modality"] == self.args.modality1].values[0, 2:]
            modality0 = torch.tensor(modality0.astype(np.float32))
            modality1 = torch.tensor(modality1.astype(np.float32))

        return {
            "modality0": modality0,
            "modality1": modality1,
            "modality": modality,
            "label": label,
            "IID": patient_id,
        }


def ADNI_ROI_collate(batch, args, task="generation"):
    assert task in ["generation", "classification"]
    batch = default_collate(batch)

    if task == "generation":
        modality0_indices, modality1_indices, paired_in_modality0, paired_in_modality1 = (
            ADNI_ROI.generate_indices(args, batch)
        )
        return batch, modality0_indices, modality1_indices, paired_in_modality0, paired_in_modality1
    else:
        indices = torch.tensor([i for i, item in enumerate(batch["label"]) if item != 3])
        for key in batch.keys():
            if isinstance(batch[key], list):
                batch[key] = [batch[key][i] for i in indices]
            else:
                batch[key] = batch[key][indices]
        modality0_indices, modality1_indices, _, _ = ADNI_ROI.generate_indices(args, batch)
        complete_indices = list(set(modality0_indices) & set(modality1_indices))
        single_modality0_indices = [i for i in modality0_indices if i not in complete_indices]
        single_modality1_indices = [i for i in modality1_indices if i not in complete_indices]

        return batch, single_modality0_indices, single_modality1_indices, complete_indices


DATASETS = {
    "brats2019": Brats2019,
    "adni_roi": ADNI_ROI,
}
CALLATE_FNC = {"adni_roi": ADNI_ROI_collate}

if __name__ == "__main__":
    pass
