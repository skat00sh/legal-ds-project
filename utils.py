import json
import random
import os
import shutil
import string
from typing import Tuple

DEBUG_FLAG = 0


def generate_random_samples(items: list, num_samples: int, random_seed: int = 42):
    random.seed(a=random_seed)
    return random.choices(items, k=num_samples)


def print_list_from_dict(item: dict(), key: string):
    for i in item:
        print(i[key])


def get_data_and_annotations(
    path: string = "./data/ldsi_w21_curated_annotations_v2.json",
) -> Tuple[list, list]:
    with open(path) as annotations:
        data = json.load(annotations)
        annotated_docs = []

        for doc_id in data["annotations"]:
            if doc_id["document"] not in annotated_docs:
                annotated_docs.append(doc_id["document"])

    return data, annotated_docs


def split_data(
    path: string = "./data/ldsi_w21_curated_annotations_v2.json",
) -> Tuple[list, list, list, list, list, list]:

    """
    Splits the dataset into 3 parts Test, Dev and Training set. See the return order carefully.

    :return
            (train_denied,
            train_granted,
            dev_denied,
            dev_granted,
            test_denied,
            test_granted,)


    """
    data, annotated_docs = get_data_and_annotations(path)

    granted = []
    denied = []
    remanded = []
    for i in data["documents"]:
        if i["_id"] in annotated_docs:
            if i["outcome"] == "granted":
                granted.append(i)
            elif i["outcome"] == "denied":
                denied.append(i)
            else:
                remanded.append(i)

    test_granted = generate_random_samples(granted, 7)
    test_denied = generate_random_samples(denied, 7)
    dev_granted = generate_random_samples(
        [i for i in granted if i not in test_granted], 7
    )
    dev_denied = generate_random_samples([i for i in denied if i not in test_denied], 7)
    train_granted = [
        i for i in granted if i not in test_granted and i not in dev_granted
    ]
    train_denied = [i for i in denied if i not in test_denied and i not in dev_denied]
    if DEBUG_FLAG:
        print("Test Granted ")
        print_list_from_dict(test_granted, "_id")
        print_list_from_dict(test_granted, "name")
        print("Test Denied")
        print_list_from_dict(test_denied, "_id")
        print_list_from_dict(test_denied, "name")
        print("Dev Granted:")
        print_list_from_dict(dev_granted, "_id")
        print_list_from_dict(dev_granted, "name")
        print("Dev Denied:")
        print_list_from_dict(dev_denied, "_id")
        print_list_from_dict(dev_denied, "name")

    return (
        train_denied,
        train_granted,
        dev_denied,
        dev_granted,
        test_denied,
        test_granted,
    )


if __name__ == "__main__":
    _, _, _, _, _, _ = split_data()
