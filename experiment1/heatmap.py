import matplotlib.pyplot as plt
import numpy as np

def heatmap_saver(x_names, y_names, arr, fp, title="", cbarlabel=""):
    fig, ax = plt.subplots()
    im = ax.imshow(arr)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(x_names)))
    ax.set_yticks(np.arange(len(y_names)))
    ax.set_xticklabels(x_names)
    ax.set_yticklabels(y_names)
    for i in range(len(y_names)):
        for j in range(len(x_names)):
            text = ax.text(j, i, arr[i,j], ha="center", va="center", color="w")

    ax.set_title(title)
    ax.set_ylabel('Code Size')
    ax.set_xlabel('Memory Size')
    fig.tight_layout()
    plt.savefig(fp)


