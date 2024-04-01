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
            Keys: client_idx numbers (0, 1, 2, ...), 'test', 'valid'
            Values:
                if key is client_idx:
                    The values are client_idx dictionary containing the patient data for each client_idx.
                    Each client_idx dictionary has the following key-value:
                    - args.modality0: A list of patients assigned to the client_idx with modality 0. if this client has no modality0, this value would be []
                    - args.modality1: A list of patients assigned to the client_idx with modality 1. if this client has no modality1, this value would be []
                    - "paired_patient": A list of patients assigned to the client_idx that have both modalities.
                    - "modality": A list of modalities assigned to the client_idx.
                if key is 'test' or 'valid':
                    value is a Dict, whose key-value are:
                    - "patient": A list of patients assigned to the test or validation set.
                    - "modality": A list of modalities assigned to the test or validation set.
        patient_path_dict: Dict
            A dictionary representing the path of each patient.
            The keys are the patient names and the values are the path of this patient.
    """

    """
    Initialize Basic Variables
    """
    # initialize patient partition
    patient_partition = {}
    for i in range(args.client_num):
        patient_partition[i] = {}
        patient_partition[i]["modality"] = []
        patient_partition[i]["paired_patient"] = []
        patient_partition[i][args.modality0] = []
        patient_partition[i][args.modality1] = []
    patient_partition["test"] = {}
    patient_partition["valid"] = {}
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

    """
    Assgin Test Set and Validation Set
    """
    patient_partition["test"]["patient"] = patient4test_list
    patient_partition["test"]["modality"] = [args.modality0, args.modality1]
    # assign validation set
    random.shuffle(patient4train_list)
    num_extract = int(len(patient4train_list) * 0.1)
    patient_partition["valid"]["patient"] = patient4train_list[:num_extract]
    patient_partition["valid"]["modality"] = [args.modality0, args.modality1]

    """
    Generate Training Set
    """
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

    # print(len(patient4train_list))
    return patient_partition, patient_path_dict


def add_argument():
    parser = argparse.ArgumentParser()
    # parameters for partitioning the patients
    parser.add_argument(
        "-rm0",
        "--ratio_m0",
        type=float,
        default=1.0,
        help="Ratio of clients having modality 0",
    )
    parser.add_argument(
        "-rm1",
        "--ratio_m1",
        type=float,
        default=1.0,
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
        default=60,
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
