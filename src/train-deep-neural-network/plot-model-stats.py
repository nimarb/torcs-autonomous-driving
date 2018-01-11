import os
import os.path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from textwrap import wrap
#matplotlib.use('pgf')


"""from: http://bkanuka.com/articles/native-latex-plots/"""
def figsize(scale):
    fig_width_pt = 512                          # Get this from LaTeX using \the\textwidth or rather \the\columnwidth for IEEE docs
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


PGF_WITH_LATEX = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    #"figure.figsize": 464,
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
                #65), item[yaxis_to_plot[1]][:65], label=labels[i])
        i += 1

    plt.legend(loc='upper center', shadow=False, fontsize='large', ncol=2)
    plt.yscale('log')
    plt.title("\n".join(wrap(
        json_strs[0][yaxis_to_plot[0]]
        + " of DNN on map: "
        + json_strs[0]["data_names"][0].replace("_", " ")
        + ".")))
    plt.ylabel(json_strs[0][yaxis_to_plot[0]])
    plt.xlabel("num_epochs".replace("_", " "))
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

    i = 0
    labs = ["angle", "distance"]
    for item in json_strs:
        #labels.append(
        #    labs[i])
        #i += 1
        labels.append(
            "architecture: "
        #    str(item["img_width"])
        #    + "x"
        #    + str(item["img_height"])
        #    + "px; opt:"
        #    + "px")
            + item["camera_perspective"].replace("_", " "))

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
    plt.savefig('{}.pgf'.format(fname_with_path))
    plt.savefig('{}.pdf'.format(fname_with_path))


if __name__ == "__main__":
    yaxis_to_plot = ["metrics", "val_mae_hist"]
    if len(sys.argv) == 1:
        labels = []
        versions = ["30567", "31159", "31785", "32414"]
        file_names = []
        for ver in versions:
            file_names.append(
                "../models/simple-distance-perspective_comp/"
                + "modelslearndrive-model-"
                + ver
                + "-metadata.json")

    if len(sys.argv) >= 2:
        file_names = sys.argv[1:]

    labels = get_img_size_labels(file_names)
    plots_train_model_json(file_names, labels, yaxis_to_plot)
