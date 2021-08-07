import numpy as np
import os
import json

def modifications(features, labels):
    def new_classes(features, labels, subtype, idx):
        subtype = subtype[idx]
        labels_right = (((features[:, 0] - 1) - (labels[idx, :, 1] - 1)) / 30)
        labels_left = (labels[idx, :, 1] - 1) / 30
        labels_left[(labels_left % 1) != 0] = labels_right[(labels_left % 1) != 0] + 20
        labels[idx, :, 1] = labels_left

    def change_classes(labels):
        washing = np.asarray([0, 30, 60, 120])
        cooking = np.asarray([0, 30, 60, 90])
        storage = np.asarray([0, 60, 60, 90, 120])

        labels[0, :, 0] = washing[labels[0, :, 0]]
        labels[1, :, 0] = cooking[labels[1, :, 0]]
        labels[2, :, 0] = storage[labels[2, :, 0]]

    washing = np.asarray([0, 30, 60, 120])
    cooking = np.asarray([0, 30, 60, 90])
    storage = np.asarray([0, 60, 60, 90, 120])
    subtype = [washing, cooking, storage]

    labels_base = np.copy(labels)
    new_classes(features, labels, subtype, 0)
    new_classes(features, labels, subtype, 1)
    new_classes(features, labels, subtype, 2)
    change_classes(labels)

    return features, labels, labels_base


def read_from_file(file, features=None, labels=None):

    if features is None:
        features = []
        labels = {"C": [], "W": [], "S": []}
    with open(file) as json_file:
        data = json.load(json_file)
        for elem in data:
            example_input = []
            nn_input = elem["input"]
            nn_output = elem["output"]
            example_input.append(nn_input["walls"][0] + 1)  # duzina zida
            for util in nn_input["utils"]:
                example_input.append(util["Pos"] + 1)  # gde su mu vent, kanal, prozor, ulaz, izlaz
            features.append(example_input)
            storage = False
            for output in nn_output:
                if output["type"] != "P":
                    labels[output["type"]].append([output["subType"], output["pos"] + 1, output["subType"]])
                    if output["type"] == "S":
                        storage = True
            if not storage:
                labels["S"].append([0, 0, 0])

        return np.asarray(features), np.asarray([labels["W"], labels["C"], labels["S"]])


def read_from_folder(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))
    features = []
    labels = {"C": [], "W": [], "S": []}
    for file in files:
        read_from_file(file, features, labels)
    return modifications(np.asarray(features), np.asarray([labels["W"], labels["C"], labels["S"]]))


def read_from_file_modif(file):
    features = []
    labels = {"C": [], "W": [], "S": []}
    features, labels = read_from_file(file, features, labels)
    return modifications(features, labels)
