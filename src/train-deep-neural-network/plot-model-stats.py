import os
import os.path
import sys

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


def plots_train_model_json(filenames, labels, yaxis_to_plot):
    """Plots data from the model metadata

    Arguments:
        filenames: list, paths of metadata files
        labels: list, legend name for each plot
        yaxis_to_plot: list, name of func to plot and value to plot"""
    matplotlib.rcParams.update(PGF_WITH_LATEX)
    json_strs = []
    json_str = {}
    for filename in filenames:
        json_str = json.load(open(filename))
        json_strs.append(json_str)

    i = 0
    for item in json_strs:
        plt.plot(
            np.arange(
                item["num_epochs"]), item[yaxis_to_plot[1]], label=labels[i])
        i += 1

    plt.legend(loc='upper center', shadow=False, fontsize='large')
    plt.yscale('log')
    plt.title("\n".join(wrap(
        json_strs[0][yaxis_to_plot[0]]
        + " of DNN on map: "
        + json_strs[0]["data_name"]
        + ".")))
    plt.ylabel(json_strs[0][yaxis_to_plot[0]])
    plt.xlabel("num_epochs")
    plt.tight_layout()
    save_plots(file_names)
    plt.show()


def get_img_size_labels(file_names):
    """Returns the image sizes in a list from a file name list

    Arguments:
        file_names: list, paths of metadata files"""
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


def save_plots(file_names):
    """Saves the current plot in the current dir

    Arguments:
        file_names: list, paths of metadata files"""
    unified_file_name_list = ["learndrive-model"]
    for name in file_names:
        fname = name.split(sep='/')[-1]
        fnumber = str(fname.split(sep='-')[-2])
        unified_file_name_list.append("-" + fnumber)

    unified_file_names = "".join(unified_file_name_list)
    fname_with_path = "/".join(file_names[0].split(sep="/")[:-1]) \
                        + "/" \
                        + unified_file_names
    #plt.savefig('{}.pgf'.format(fname_with_path))
    plt.savefig('{}.pdf'.format(fname_with_path))


if __name__ == "__main__":
    yaxis_to_plot = ["loss_function", "loss_hist"]
    if len(sys.argv) == 1:
        labels = []
        versions = ["48493", "52163", "61302"]
        file_names = []
        for ver in versions:
            file_names.append(
                "../models/learndrive-model-" + ver + "-metadata.json")

    if len(sys.argv) >= 2:
        file_names = sys.argv[1:]

    labels = get_img_size_labels(file_names)
    plots_train_model_json(file_names, labels, yaxis_to_plot)
