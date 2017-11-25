import os
import os.path

import numpy as np
import matplotlib.pyplot as plt
import json


def plot_model_json(filename):
    json_str = {}
    json_str = json.load(open(filename))
    loss_hist = json_str["loss_hist"]
    plt.plot(np.arange(json_str["num_epochs"]), loss_hist)
    plt.yscale('log')
    plt.title(
        json_str["loss_function"]
        + " of CNN using the "
        + json_str["data_name"]
        + " dataset."
        + " Image size: "
        + str(json_str["img_width"])
        + " by "
        + str(json_str["img_height"])
        + " pixels.")
    plt.ylabel(json_str["loss_function"])
    plt.xlabel("num_epochs")
    plt.show()


if __name__ == "__main__":
    version = "10794"
    file_name = "../models/learndrive-model-" + version + "-metadata.json"
    plot_model_json(file_name)

