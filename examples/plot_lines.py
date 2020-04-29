import numpy as np

from paper_plot import Plotter


def get_dummy_data():
    x = np.arange(100)
    y = np.random.random(len(x))
    return x, y


if __name__ == "__main__":
    x, y = get_dummy_data
    plotter = Plotter(save_dir="data")
    plotter.plot_line(x, y)
