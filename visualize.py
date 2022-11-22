import matplotlib.pylab as plt
import numpy as np

def plot_reconstruction(datalr, datasr, datahr=None, vm=1, nsub=2, cmap="afmhot"):
    """Plot the dirty image, POLISH reconstruction,
    and (optionally) the high resolution true sky image
    """
    vminlr = 0
    vmaxlr = 22500
    vminsr = 0
    vmaxsr = 22500
    vminhr = 0
    vmaxhr = 22500
    if datahr is None:
        num_plots = 2
        fig = plt.figure(figsize=(10, 6))
    else:
        num_plots = 3
        fig = plt.figure(figsize=(13, 6))

    ax1 = plt.subplot(1, num_plots, 1)
    plt.title("Dirty map", color="C1", fontsize=17)
    plt.axis("off")
    plt.imshow(
        np.squeeze(datalr),
        cmap=cmap,
        vmax=vmaxlr,
        vmin=vminlr,
        aspect="auto",
        extent=[0, 1, 0, 1],
    )
    plt.setp(ax1.spines.values(), color="C1")

    ax2 = plt.subplot(1, num_plots, 2, sharex=ax1, sharey=ax1)
    plt.title("POLISH reconstruction", c="C2", fontsize=17)
    plt.imshow(
        np.squeeze(datasr),
        cmap=cmap,
        vmax=vmaxsr,
        vmin=vminsr,
        aspect="auto",
        extent=[0, 1, 0, 1],
    )
    plt.axis("off")

    if num_plots == 3:
        ax3 = plt.subplot(1, num_plots, 3, sharex=ax1, sharey=ax1)
        plt.title("True sky", c="k", fontsize=17)
        plt.imshow(
            np.squeeze(datahr),
            cmap=cmap,
            vmax=vmaxsr,
            vmin=vminsr,
            aspect="auto",
            extent=[0, 1, 0, 1],
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()