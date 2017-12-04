import os
import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from textwrap import wrap


PGF_WITH_LATEX = {
    #"pgf.texsystem": "pdflatex",
    #"text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    #"figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    #"figure.figsize": 1,
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }


def plots_train_model_json(filenames, labels):
    matplotlib.rcParams.update(PGF_WITH_LATEX)
    json_strs = []
    json_str = {}
    for filename in filenames:
        json_str = json.load(open(filename))
        json_strs.append(json_str)

    i = 0
    for item in json_strs:
        plt.plot(
            np.arange(item["num_epochs"]), item["loss_hist"], label=labels[i])
        i += 1

    legend = plt.legend(loc='upper center', shadow=False, fontsize='large')

    plt.yscale('log')
    plt.title("\n".join(wrap(
        json_strs[0]["loss_function"]
        + " of DNN on map: "
        + json_strs[0]["data_name"]
        + ".")))
    plt.ylabel(json_strs[0]["loss_function"])
    plt.xlabel("num_epochs")
    plt.tight_layout()
    save_fig(file_names[0])
    plt.show()


def get_img_size_labels(file_names):
    labels = []
    json_strs = []
    json_str = {}
    for filename in file_names:
        json_str = json.load(open(filename))
        json_strs.append(json_str)
    
    for item in json_strs:
        labels.append(
            str(item["img_width"])
            + "x"
            + str(item["img_height"])
            + "px")

    return labels

def plot_model_json(filename):
    matplotlib.rcParams.update({'font.size': 14})
    json_str = {}
    json_str = json.load(open(filename))
    loss_hist = json_str["loss_hist"]
    mae_hist = json_str["train_mae_hist"]
    plt.plot(np.arange(json_str["num_epochs"]), loss_hist)
    plt.plot(np.arange(json_str["num_epochs"]), mae_hist)
    plt.yscale('log')
    plt.title("\n".join(wrap(
        json_str["loss_function"]
        + " of CNN on map: "
        + json_str["data_name"]
        + ". "
        + "Image size: "
        + str(json_str["img_width"])
        + " by "
        + str(json_str["img_height"])
        + "px.")))
    plt.ylabel(json_str["loss_function"])
    plt.xlabel("num_epochs")
    plt.tight_layout()
    save_fig(file_name)
    plt.show()


def save_fig(filename):
    #plt.savefig('{}.pgf'.format(filename))
    plt.savefig('{}.pdf'.format(filename))


if __name__ == "__main__":
    labels = []
    versions = ["48493", "52163", "61302"]
    file_names = []
    for ver in versions:
        file_names.append(
            "../models/learndrive-model-" + ver + "-metadata.json")

    labels = get_img_size_labels(file_names)

    plots_train_model_json(file_names, labels)

