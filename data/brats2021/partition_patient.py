import argparse
import os
import random
from typing import Dict, List, Union


def partition_patient(args) -> Dict[int, Dict[Union[int, str], List[str]]]:
    """
    Preprocesses the data by partitioning the patients into different clients based on the given arguments.

    Args:
        args (object): An object containing the input arguments.

    Returns:
        patient_partition: dict
            A dictionary representing the patient partition.
            The keys are the client numbers (0, 1, 2, ...) and the values are client dictionary containing the patient data for each client.
            Each client dictionary has the following keys:
            - 0: A list of patients assigned to the client with modality 0.
            - 1: A list of patients assigned to the client with modality 1.
            - "paired_patient": A list of patients assigned to the client that have both modalities. (if the client has single modality, this key does not exist.)
    """
    # initialize patient partition
    patient_partition = {}
    for i in range(args.client_num):
        patient_partition[i] = {}

    patient = os.listdir(os.path.join(args.data_path, "train"))
    # randomly split the patients into train and test
    random.shuffle(patient)
    train_patient = patient[: int(len(patient) * args.ratio_train)]
    test_patient = patient[int(len(patient) * args.ratio_train) :]
    # randomly sample clients from {0,1,...,client_num-1} to be the clients that have modality 0 and clients that have modality 1
    client_m0_num = int(args.client_num * args.ratio_m0)
    client_m1_num = int(args.client_num * args.ratio_m1)
    client_m0 = set(random.sample(range(args.client_num), client_m0_num))
    client_m1 = set(range(args.client_num)) - client_m0
    client_m1.update(
        random.sample(client_m0, client_m1_num - args.client_num + client_m0_num)
    )  # randomly sample rest client_m2 from client_m1
    client_both = client_m1.intersection(client_m0)  # clients that have both modalities
    # the number of patients having modality 0 on each client in client_m1
    patient_num_client_m0 = int(len(train_patient) / client_m0_num)
    patient_num_client_m1 = int(len(train_patient) / client_m1_num)
    # number of paired data in each client that has both modalities
    min_patient_num = min(patient_num_client_m0, patient_num_client_m1)
    min_patient_index = 0 if min_patient_num == patient_num_client_m0 else 1
    pari_num = int(min_patient_num * args.ratio_pair)
    # averagely distribute the patients to the each client in client_m0
    for i, client in enumerate(client_m0):
        patient_partition[client][0] = train_patient[
            i * patient_num_client_m0 : (i + 1) * patient_num_client_m0
        ]
    patient = set(patient)
    # assign the paired patients to the clients_both with modality 1
    for client in client_both:
        paired_patient = random.sample(patient_partition[client][0], pari_num)
        patient_partition[client][1] = paired_patient
        patient_partition[client]["paired_patient"] = paired_patient
        patient = patient - set(paired_patient)
    # assign the unpaired patients to the clients_both with modality 1
    for client in client_both:
        unpaired_patient_m1 = random.sample(
            patient - set(patient_partition[client][0]),
            patient_num_client_m1 - pari_num,
        )
        patient_partition[client][1] = (
            patient_partition[client][1] + unpaired_patient_m1
        )
    # assign the patients to the clients_m1 with modality 1
    for client in client_m1 - client_both:
        patient_partition[client][1] = random.sample(patient, patient_num_client_m1)

    return patient_partition


def add_argument():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--data_path", type=str, default="raw", help="Path to the data")
    parser.add_argument(
        "-m",
        "--modality",
        choices=["t1", "t2", "flair", "mri"],
        default=["t1", "t2"],
        help="Modalities to preprocess",
    )
    args = parser.parse_args()
    if (
        int(args.ratio_m0 * args.client_num) + int(args.ratio_m1 * args.client_num)
        < args.client_num
    ):
        raise ValueError(
            "The sum of the number of client_m1 and the number of client_m2 must >= client_num."
        )
    return args


if __name__ == "__main__":
    args = add_argument()
    random.seed(args.seed)
    patient_partition = partition_patient(args)
