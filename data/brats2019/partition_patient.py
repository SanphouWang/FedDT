import argparse
import os
import random
from typing import Dict, List, Union, Set
import json
import pickle


def partition_patient(args) -> Dict[int, Dict[str, List[str]]] and List[str] and Dict[str, str]:
    """
    Preprocesses the data by partitioning the patients into different clients based on the given arguments.

    Args:
        args (object): An object containing the input arguments.

    Returns:
        patient_partition: Dict
            A dictionary representing the patient partition.
            The keys are the client_idx numbers (0, 1, 2, ...) and the values are client_idx dictionary containing the patient data for each client_idx.
            Each client_idx dictionary has the following keys:
            - args.modality0: A list of patients assigned to the client_idx with modality 0.
            - args.modality1: A list of patients assigned to the client_idx with modality 1.
            - "paired_patient": A list of patients assigned to the client_idx that have both modalities. (if the client_idx has single modality, this key does not exist.)
            - "modality": A list of modalities assigned to the client_idx.
        patient4test_list: List
            A list of patients assigned to the test set.
        patient_path_dict: Dict
            A dictionary representing the path of each patient.
            The keys are the patient names and the values are the path of each patient.
    """
    # initialize patient partition
    patient_partition = {}
    for i in range(args.client_num):
        patient_partition[i] = {}
    patient_partition["test"] = {}
    HGG_patient_list = os.listdir(os.path.join(args.data_path, "HGG"))
    LGG_patient_list = os.listdir(os.path.join(args.data_path, "LGG"))
    all_patient_list = HGG_patient_list + LGG_patient_list
    patient_path_dict = {}
    for patient in all_patient_list:
        patient_path_dict[patient] = (
            os.path.join(args.data_path, "HGG", patient)
            if patient in HGG_patient_list
            else os.path.join(args.data_path, "LGG", patient)
        )
    random.shuffle(all_patient_list)
    patient4train_list = all_patient_list[: int(len(all_patient_list) * args.ratio_train)]
    patient4test_list = all_patient_list[int(len(all_patient_list) * args.ratio_train) :]
    # randomly sample clients from {0,1,...,client_num-1} to be the clients that have modality 0 and clients that have modality 1
    client_m0_num = int(args.client_num * args.ratio_m0)
    client_m1_num = int(args.client_num * args.ratio_m1)
    client_m0 = set(random.sample(range(args.client_num), client_m0_num))
    client_m1 = set(range(args.client_num)) - client_m0
    client_m1.update(
        random.sample(client_m0, client_m1_num - args.client_num + client_m0_num)
    )  # randomly sample rest client_m2 from client_m1
    client_both = client_m1.intersection(client_m0)  # clients that have both modalities
    # the number of patients having modality 0 on each client_idx in client_m1
    patient_num_client_m0 = int(len(patient4train_list) / client_m0_num)
    patient_num_client_m1 = int(len(patient4train_list) / client_m1_num)
    # number of paired data in each client_idx that has both modalities
    min_patient_num = min(patient_num_client_m0, patient_num_client_m1)
    min_patient_index = 0 if min_patient_num == patient_num_client_m0 else 1
    pari_num = int(min_patient_num * args.ratio_pair)
    # averagely distribute the patients to the each client_idx in client_m0
    for i, client_idx in enumerate(client_m0):
        patient_partition[client_idx][args.modality0] = patient4train_list[
            i * patient_num_client_m0 : (i + 1) * patient_num_client_m0
        ]
        patient_partition[client_idx]["modality"] = [args.modality0]
    # assign the paired patients to the clients_both with modality 1
    for client_idx in client_both:
        paired_patient = random.sample(patient_partition[client_idx][args.modality0], pari_num)
        patient_partition[client_idx][args.modality1] = paired_patient
        patient_partition[client_idx]["paired_patient"] = paired_patient
        patient4train_list = list(set(patient4train_list) - set(paired_patient))
        patient_partition[client_idx]["modality"] = [args.modality0, args.modality1]

    # assign the unpaired patients to the clients_both with modality 1
    for client_idx in client_both:
        unpaired_patient_m1 = random.sample(
            set(patient4train_list) - set(patient_partition[client_idx][args.modality0]),
            patient_num_client_m1 - pari_num,
        )
        patient_partition[client_idx][args.modality1] = (
            patient_partition[client_idx][args.modality1] + unpaired_patient_m1
        )
        patient4train_list = list(set(patient4train_list) - set(unpaired_patient_m1))

    # assign the patients to the clients_m1 with modality 1
    for client_idx in client_m1 - client_both:
        patient_partition[client_idx][args.modality1] = random.sample(
            patient4train_list, patient_num_client_m1
        )
        patient_partition[client_idx]["modality"] = [args.modality1]
        patient4train_list = list(
            set(patient4train_list) - set(patient_partition[client_idx][args.modality1])
        )
    # assgin test set
    patient_partition["test"][args.modality0] = patient4test_list
    patient_partition["test"][args.modality1] = patient4test_list
    patient_partition["test"]["modality"] = [args.modality0, args.modality1]
    # print(len(patient4train_list))
    return patient_partition, patient_path_dict


def add_argument():
    parser = argparse.ArgumentParser()
    # parameters for partitioning the patients
    parser.add_argument(
        "-rm0",
        "--ratio_m0",
        type=float,
        default=0.6,
        help="Ratio of clients having modality 0",
    )
    parser.add_argument(
        "-rm1",
        "--ratio_m1",
        type=float,
        default=0.6,
        help="Ratio of clients having modality 1",
    )
    parser.add_argument(
        "-rtr", "--ratio_train", type=float, default=0.8, help="Ratio of training set"
    )
    parser.add_argument(
        "-rpr",
        "--ratio_pair",
        type=float,
        default=0.3,
        help="Ratio of paired data in the clients that have both modalities",
    )
    parser.add_argument("-cn", "--client_num", type=int, default=10)
    parser.add_argument("-s", "--seed", type=int, default=42)
    # parameters for the data description
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/hardDisk1/wwmm/sanphou/FedDT/data/brats2019/",
        help="Path to the data. Recommand to use the absolute path.",
    )
    parser.add_argument(
        "--modality0", "-m0", type=str, choices=["flair", "t1", "t2", "t1ce"], default="t1"
    )
    parser.add_argument(
        "--modality1", "-m1", type=str, choices=["flair", "t1", "t2", "t1ce"], default="t2"
    )
    parser.add_argument(
        "--slice_idx_begin",
        type=int,
        default=50,
        help="beginning slice index, each volume in Brats2019 has 155 slices",
    )
    parser.add_argument(
        "--slice_idx_end",
        type=int,
        default=80,
        help="ending slice index, each volume in Brats2019 has 155 slices",
    )
    args = parser.parse_args()
    if (
        int(args.ratio_m0 * args.client_num) + int(args.ratio_m1 * args.client_num)
        < args.client_num
    ):
        raise ValueError(
            "The sum of the number of client_m1 and the number of client_m2 must >= client_num."
        )

    if args.modality0 == args.modality1:
        raise ValueError("The two modalities must be different.")
    return args


if __name__ == "__main__":
    args = add_argument()
    random.seed(args.seed)
    patient_partition, patient_path = partition_patient(args)

    if not os.path.exists("preprocessed_files"):
        os.makedirs("preprocessed_files")
    with open("preprocessed_files/args.pkl", "wb") as f:
        pickle.dump(vars(args), f)

    with open("preprocessed_files/patient_partition.pkl", "wb") as f:
        pickle.dump(patient_partition, f)

    with open("preprocessed_files/patient_path.pkl", "wb") as f:
        pickle.dump(patient_path, f)
