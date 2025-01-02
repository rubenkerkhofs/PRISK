import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_assets_and_basins(assets_with_basins, basins_processed, save=False):
    basins_processed.plot(color=basins_processed.color, figsize=(20, 20))
    plt.scatter(
        assets_with_basins.Longitude, assets_with_basins.Latitude, c="red", s=50
    )
    if save:
        plt.savefig("map.png", transparent=True)
