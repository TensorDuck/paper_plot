import numpy as np

from paper_plot import Plotter


def get_dummy_data():
    x = np.arange(100) - 50
    y = (np.random.random(len(x)) - 0.5) * 100
    return x, y


if __name__ == "__main__":
    x, y = get_dummy_data()
    plotter = Plotter(save_dir="data")
    plotter.set_pnas_dimensions()
    plotter.plot_scatter(x, y, savename="test")
