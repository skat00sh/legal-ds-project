import json
import random
import os
import shutil
import string


def generate_random_samples(items: list, num_samples: int, random_seed: int = 42):
    random.seed(a=random_seed)
    return random.choices(items, k=num_samples)


def print_list_from_dict(item: dict(), key: string):
    for i in item:
        print(i[key])


def split_data(path: string = "./data/ldsi_w21_curated_annotations_v2.json"):
    with open(path) as annotations:
        data = json.load(annotations)
        annotated_docs = []

        for doc_id in data["annotations"]:
            if doc_id["document"] not in annotated_docs:
                annotated_docs.append(doc_id["document"])

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
        dev_denied = generate_random_samples(
            [i for i in denied if i not in test_denied], 7
        )
        print("Test Granted ")
        print_list_from_dict(test_granted, "_id")
        print_list_from_dict(test_granted, "name")
        print(
            f"Test Denied : {print_list_from_dict(test_denied, '_id')} \t {print_list_from_dict(test_denied, 'name')}\n \\"
            f"Dev Granted : {print_list_from_dict(dev_granted, '_id')} \t {print_list_from_dict(dev_granted, 'name')}\n \\"
            f"Dev Denied: {print_list_from_dict(dev_denied, '_id')} \t {print_list_from_dict(dev_denied, 'name')}\n"
        )
        return


if __name__ == "__main__":
    split_data()
