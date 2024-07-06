from itertools import product
import pickle
import sys
import os
import random
from pathlib import Path
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
import time
from typing import Dict, List, OrderedDict
from matplotlib import pyplot as plt
import pandas as pd

import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
from rich.console import Console
from rich.progress import track
from multiprocessing import Manager
from ray.util.multiprocessing import Pool
import torch.nn as nn

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())

from data.utils.datasets import CALLATE_FNC, DATASETS
from data.adni_roi.partition_patient import (
    add_argument,
    partition_patient,
    statistic_patient,
    plot_bar,
)

from src.utils.tools import (
    OUT_DIR,
    Logger,
    fix_random_seed,
    get_best_device,
    update_args_from_dict,
    move2cpu,
    move2device,
)
from src.model.model_tools import get_model_arch
from src.client.fedavg2 import FedAvgClient
from src.server.centralized import Centralized
from src.server.fedavg2 import get_fedavg_argparser, local_time
from multiprocessing import Pool
from multiprocessing import Pool


def final_df(k):
    path = f"/mnt/hardDisk1/wwmm/sanphou/FedDT/out/Centralized/adni_roi/{begin_time}/classification/k={k}"
    final_df_dict = {
        "$r_{both}$": [],
        "$\lambda$": [],
        "generator": [],
        "Precision1": [],
        "Recall1": [],
        "F11": [],
        "Accuracy1": [],
        "Precision2": [],
        "Recall2": [],
        "F12": [],
        "Accuracy2": [],
        "Precision3": [],
        "Recall3": [],
        "F13": [],
        "Accuracy3": [],
    }
    for ratio_both in [0.2, 0.5, 0.8]:
        for sigma in [0.2, 0.5, 0.8]:
            dir_path = os.path.join(path, f"ratio_both={ratio_both}/sigma={sigma}/")
            file_names = os.listdir(dir_path)
            file_names.sort()
            for file_name in file_names:
                if file_name == "without_generator":
                    generator = "None"
                else:
                    generator = file_name
                final_df_dict["$r_{both}$"].append(ratio_both)
                final_df_dict["$\lambda$"].append(sigma)
                final_df_dict["generator"].append(generator)

                metric_path = os.path.join(dir_path, file_name, "metrics.csv")
                df = pd.read_csv(metric_path, index_col=None)
                all_columns = df.columns.tolist()
                if "Unnamed: 0" in all_columns:
                    all_columns.remove("Unnamed: 0")
                for test_set_index, test_set in enumerate(df.index):
                    for metric in all_columns:
                        final_df_dict[f"{metric}{test_set_index+1}"].append(
                            df.loc[test_set, metric]
                        )
    final_df = pd.DataFrame(final_df_dict)

    def format_to_two_decimals(x):
        if isinstance(x, (int, float)):
            return f"{x:.2f}"
        return x

    final_df = final_df.applymap(format_to_two_decimals)
    final_df.to_csv(
        f"/mnt/hardDisk1/wwmm/sanphou/FedDT/out/Centralized/adni_roi/{begin_time}/classification/k={k}/final_metrics.csv",
        index=False,
    )


def aggregate_df():
    dataframes = []
    for k in [1, 2, 3, 4, 5]:
        path = f"/mnt/hardDisk1/wwmm/sanphou/FedDT/out/Centralized/adni_roi/{begin_time}/classification/k={k}/final_metrics.csv"
        dataframes.append(pd.read_csv(path))
    hyperparameter = dataframes[0][["$r_{both}$", "$\lambda$", "generator"]]
    stacked_array = np.dstack([df.iloc[:, 3:].values for df in dataframes])

    mean_array = np.mean(stacked_array, axis=2)
    std_array = np.std(stacked_array, axis=2)

    mean_df = pd.DataFrame(mean_array, columns=dataframes[0].columns[3:]).round(2)
    std_df = pd.DataFrame(std_array, columns=dataframes[0].columns[3:]).round(2)

    def format_to_two_decimals(x):
        if isinstance(x, (int, float)):
            return f"{x:.2f}"
        return x

    mean_df = mean_df.applymap(format_to_two_decimals)
    std_df = std_df.applymap(format_to_two_decimals)
    mean_std_df = mean_df.astype(str) + "$\pm$" + std_df.astype(str)
    mean_std_df = pd.concat([hyperparameter, mean_std_df], axis=1)
    mean_std_df.to_csv(
        f"/mnt/hardDisk1/wwmm/sanphou/FedDT/out/Centralized/adni_roi/{begin_time}/classification/mean_std_metrics_{local_time()}.csv",
        index=False,
    )


def process(params):
    k, ratio_both, sigma, lambda_paired = params
    if round_idx == 0:
        generation_models = ["cgan_p2p", "pixel2pixel", "cyclegan"]
        classification_methods = ["none", "cgan_p2p", "pixel2pixel", "cyclegan"]
    else:
        generation_models = ["cgan_p2p"]
        classification_methods = ["cgan_p2p"]

    """
    Generate Data Partition
    """
    sleep_time = random.uniform(0, 4)
    time.sleep(sleep_time)
    data_args = add_argument()
    data_args.k = k
    data_args.ratio_both = ratio_both
    data_args.sigma = sigma
    patient_partition = partition_patient(data_args)
    preprocessed_file_directory = f"{begin_time}/k={k}/ratio_both={ratio_both}/sigma={sigma}"
    data_args.preprocessed_file_directory = preprocessed_file_directory
    data_args.class_num = 3
    if not os.path.exists(os.path.join(data_args.data_path, preprocessed_file_directory)):
        os.makedirs(os.path.join(data_args.data_path, preprocessed_file_directory))
    with open(
        os.path.join(data_args.data_path, preprocessed_file_directory, "args.pkl"),
        "wb",
    ) as f:

        pickle.dump(vars(data_args), f)
    with open(
        os.path.join(data_args.data_path, preprocessed_file_directory, "patient_partition.pkl"),
        "wb",
    ) as f:
        pickle.dump(patient_partition, f)
    statistic_dict = statistic_patient(patient_partition, data_args)
    plot_bar(statistic_dict, data_args)
    ratio_both = data_args.ratio_both
    sigma = data_args.sigma
    args = get_fedavg_argparser().parse_args()
    """
    Generation
    """
    args.preprocessed_file_directory = preprocessed_file_directory
    args.task = "generation"
    args.out_dir = f"{begin_time}/generation/k={k}/ratio_both={ratio_both}/sigma={sigma}"
    args.gen_resume = ""
    args.lambda_paired = lambda_paired

    for generation_model in generation_models:
        args.gen_model = generation_model
        args.log_name = generation_model

        centralized = Centralized(args=deepcopy(args))
        centralized.generation_workflow()
    """
    Classification
    """
    # args.class_epochs =
    args.task = "classification"
    args.out_dir = f"{begin_time}/classification/k={k}/ratio_both={ratio_both}/sigma={sigma}"
    for classification_method in classification_methods:
        if classification_method == "none":
            args.use_generator = False
            args.log_name = "without_generator"
            centralized = Centralized(args=deepcopy(args))
            centralized.classification_workflow()
        else:
            args.gen_model = classification_method
            args.use_generator = True
            args.log_name = generation_model
            args.gen_resume = os.path.join(
                OUT_DIR,
                "Centralized",
                args.dataset,
                f"{begin_time}/generation/k={k}/ratio_both={ratio_both}/sigma={sigma}/{generation_model}/checkpoint/checkpoint.pt",
            )
            centralized = Centralized(args=deepcopy(args))
            centralized.classification_workflow()


if __name__ == "__main__":
    random.seed(42)
    # begin_time = "2024-07-03-12:24:46"
    multiprocess = True
    round_idx = 1
    if multiprocess:
        for lambda_paired in [500]:
            # begin_time = local_time()
            begin_time = "2024-07-04-09:31:46"
            pool = Pool(processes=30)
            params = product([1, 2, 3, 4, 5], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [lambda_paired])
            pool.map(process, params)
            pool.close()
            pool.join()
            for k in [1, 2, 3, 4, 5]:
                final_df(k)
            aggregate_df()
            # round_idx += 1
    else:
        begin_time = local_time()
        params = [(3, 0.5, 0.5, 100)]
        for param in params:
            process(param)
