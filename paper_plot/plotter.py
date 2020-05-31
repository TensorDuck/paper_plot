from pathlib import Path  # great for forming paths that are OS agnostic

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_COLOR_SEQUENCE = ["b", "r", "g", "m", "c", "y"]


class Plotter(object):
    # class defaults
    _default_color_sequence = ["b", "r", "g", "m", "c", "y"]
    _default_line_types = ["-", "--", ":", "-."]
    _alphabet_sequence = "abcdefghijklmnopqrstuvwxyz"

    def __init__(
        self,
        save_dir: str = "test",
        color_sequence: list = _default_color_sequence,
        line_types: list = _default_line_types,
    ):
        """
        Object to track save directory and make figures

        The save directory is set on initialize, but can be changed live when
        needed. The object also provides convenience methods for quickly making
        figures to plot. The goal is to strike a balance with making consistent
        figures for a paper/presentation for the most common figures I
        encountered, while also providing some flexibility with what data is
        being plotted.

        Thus, this is cannot possibly provide a complete list of all possible
        figures you are likely to encounter in any one student's graduate school
        experience. Instead, it will hopefully provide a good scaffold of
        patterns on which to build your custom figure plotting module, while
        also giving a few useful methods that should help.

        Apologies for using Imperial units. If this code is still around when
        the United States switches en masse to metric, I promise to fix this
        travesty.

        Params: save_dir: relative path to directory you want to save figures.
            color_sequence: sequence of colors to use for plots line_types:
            sequence of line types to use for plots
        """
        # set the absolute path autoamtically in Unix or Windows
        self.cwd = Path(".").absolute()
        self.save_dir = self.cwd / save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.colors = color_sequence
        self.colors_black = ["k"] + self.colors

        self.line_types = line_types

        self.alphabet = list(self._alphabet_sequence)

        for col in self.colors:
            self.colors_black.append(col)

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value):
        """Set the save_dir path as an absolute path"""
        self._save_dir = Path(value).absolute()

    def save_figure(self, fig, savefile):
        """
        Save in pdf and png format

        PDF is a great format for importing directly into LaTeX. PNG is a great
        format for when PDFs dont work (i.e. presentations)
        """
        fig.savefig(
            self.save_dir / f"{savefile}.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=300,
        )

        fig.savefig(
            self.save_dir / f"{savefile}.png",
            format="png",
            bbox_inches="tight",
            dpi=300,
        )

    def set_presentation_dimensions(self):
        """
        Dimension for a presentation, i.e. on google slides in wide format
        """
        # Dimensions for laying out axes on a figure
        self.column_width_inches = 5.5
        self.double_column_width_inches = 7.0
        self.standard_box_height_inches = self.column_width_inches * (5.0 / 6.0)
        self.padding_standard = 0.6
        self.padding_standard_width_inches = 0.5  # padding on the left for y-axis label
        self.padding_standard_height_inches = (
            0.5  # padding on the bottom for x-axis label
        )
        self.padding_title_height_inches = 0.40  # padding on top for title
        self.padding_empty_inches = 0.16  # padding if no labels or tick marks exist
        self.padding_buffer_inches = 0.1  # extra padding between adjacent figures
        self.padding_buffer_xaxis = 0.3
        self.padding_buffer_yaxis = 0.3

        self.standard_font_size = 18
        self.standard_special_font_size = 27
        self.standard_thinline = 2
        self.standard_thickline = 4

        ## Matplotlib plot formats
        matplotlib.rcParams.update({"font.size": self.standard_font_size})
        matplotlib.rcParams.update({"axes.labelsize": self.standard_font_size})

        matplotlib.rcParams.update({"axes.labelpad": 4.0})
        matplotlib.rcParams.update({"legend.fontsize": 16})

        matplotlib.rcParams.update({"xtick.labelsize": 12})
        matplotlib.rcParams.update({"xtick.major.size": 3.5})
        matplotlib.rcParams.update({"xtick.minor.size": 2.0})
        matplotlib.rcParams.update({"xtick.major.pad": 3.5})

        matplotlib.rcParams.update({"ytick.labelsize": 12})
        matplotlib.rcParams.update({"ytick.major.size": 3.5})
        matplotlib.rcParams.update({"ytick.minor.size": 2.0})
        matplotlib.rcParams.update({"ytick.major.pad": 3.5})

        matplotlib.rcParams.update({"lines.linewidth": 2})
        matplotlib.rcParams.update({"patch.linewidth": 1})
        matplotlib.rcParams.update({"axes.linewidth": 1})

    def set_acs_dimensions(self):
        """
        Dimension for an acs paper
        """
        # Dimensions for laying out axes on a figure
        self.column_width_inches = 3.25  # width of column in a two-column article
        self.double_column_width_inches = 7.0  # width of a two-column spanning figure
        self.standard_box_height_inches = self.column_width_inches * (5.0 / 6.0)
        self.padding_standard = 0.4
        self.padding_standard_width_inches = (
            0.32  # padding on the left for y-axis label
        )
        self.padding_standard_width_label_inches = 0.2  # padding on the left for the y-axis when it is labeled but no tick-labels
        self.padding_standard_height_inches = (
            0.25  # padding on the bottom for x-axis label
        )
        self.padding_title_height_inches = 0.20  # padding on top for title
        self.padding_empty_inches = 0.08  # padding if no labels or tick marks exist
        self.padding_buffer_inches = 0.1  # extra padding between adjacent figures
        self.padding_buffer_xaxis = 0.15  # extra padding along vertical dimension
        self.padding_buffer_yaxis = 0.15  # extra padding along horizontal dimension

        self.standard_font_size = 9
        self.standard_special_font_size = 13
        self.standard_thinline = 1
        self.standard_thickline = 2

        ## Matplotlib plot formats
        matplotlib.rcParams.update({"font.size": self.standard_font_size})
        matplotlib.rcParams.update({"axes.labelsize": self.standard_font_size})

        matplotlib.rcParams.update({"axes.labelpad": 2.0})
        matplotlib.rcParams.update({"legend.fontsize": 8})

        matplotlib.rcParams.update({"xtick.labelsize": 6})
        matplotlib.rcParams.update({"xtick.major.size": 1.75})
        matplotlib.rcParams.update({"xtick.minor.size": 1.0})
        matplotlib.rcParams.update({"xtick.major.pad": 1.75})

        matplotlib.rcParams.update({"ytick.labelsize": 6})
        matplotlib.rcParams.update({"ytick.major.size": 1.75})
        matplotlib.rcParams.update({"ytick.minor.size": 1.0})
        matplotlib.rcParams.update({"ytick.major.pad": 1.75})

        matplotlib.rcParams.update({"lines.linewidth": 1})
        matplotlib.rcParams.update({"patch.linewidth": 0.5})
        matplotlib.rcParams.update({"axes.linewidth": 0.5})

    def set_pnas_dimensions(self):
        self.column_width_inches = 3.42  # width of column in a two-column article
        self.double_column_width_inches = 7.0  # width of a two-column spanning figure
        self.standard_box_height_inches = self.column_width_inches * (5.0 / 6.0)
        self.padding_standard = 0.4
        self.padding_standard_width_inches = (
            0.32  # padding on the left for y-axis label
        )
        self.padding_standard_width_label_inches = 0.2  # padding on the left for the y-axis when it is labeled but no tick-labels
        self.padding_standard_height_inches = (
            0.25  # padding on the bottom for x-axis label
        )
        self.padding_title_height_inches = 0.20  # padding on top for title
        self.padding_empty_inches = 0.08  # padding if no labels or tick marks exist
        self.padding_buffer_inches = 0.1  # extra padding between adjacent figures
        self.padding_buffer_xaxis = 0.15  # extra padding along vertical dimension
        self.padding_buffer_yaxis = 0.15  # extra padding along horizontal dimension

        self.standard_font_size = 9
        self.standard_special_font_size = 13
        self.standard_thinline = 1
        self.standard_thickline = 2

        matplotlib.rcParams.update({"font.size": self.standard_font_size})
        matplotlib.rcParams.update({"axes.labelsize": self.standard_font_size})

        matplotlib.rcParams.update({"axes.labelpad": 2.0})
        matplotlib.rcParams.update({"legend.fontsize": 8})

        matplotlib.rcParams.update({"xtick.labelsize": 6})
        matplotlib.rcParams.update({"xtick.major.size": 1.75})
        matplotlib.rcParams.update({"xtick.minor.size": 1.0})
        matplotlib.rcParams.update({"xtick.major.pad": 1.75})

        matplotlib.rcParams.update({"ytick.labelsize": 6})
        matplotlib.rcParams.update({"ytick.major.size": 1.75})
        matplotlib.rcParams.update({"ytick.minor.size": 1.0})
        matplotlib.rcParams.update({"ytick.major.pad": 1.75})

        matplotlib.rcParams.update({"lines.linewidth": 1})
        matplotlib.rcParams.update({"patch.linewidth": 0.5})
        matplotlib.rcParams.update({"axes.linewidth": 0.5})

    def get_single_axes(self):
        fig_width_inches = self.column_width_inches
        fig_height_inches = self.column_width_inches * (5.0 / 6.0)
        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))
        left_padding_inches = self.padding_standard_width_inches
        right_padding_inches = self.padding_empty_inches
        top_padding_inches = self.padding_empty_inches
        lower_padding_inches = self.padding_standard_height_inches

        axes_width_inches = (
            fig_width_inches - left_padding_inches - right_padding_inches
        )
        axes_height_inches = (
            fig_height_inches - top_padding_inches - lower_padding_inches
        )

        left = left_padding_inches / fig_width_inches
        bottom = lower_padding_inches / fig_height_inches

        axes_width = axes_width_inches / fig_width_inches
        axes_height = axes_height_inches / fig_height_inches

        ax_plot = fig.add_axes([left, bottom, axes_width, axes_height])

        return ax_plot, fig

    def plot_scatter(
        self, x, y, savename="plot", xname="data", yname="count", axis=None
    ):
        ax_plot, fig = self.get_single_axes()

        ax_plot.scatter(x, y)

        ax_plot.set_xlabel(xname)
        ax_plot.set_ylabel(yname)
        if axis is not None:
            ax_plot.axis(axis)
        else:
            ax_plot.axis([np.min(x), np.max(x), np.min(y), np.max(y)])

        # draw a horizontal and vertical line through the origin (0,0)
        ymin, ymax = ax_plot.get_ylim()
        xmin, xmax = ax_plot.get_xlim()
        ax_plot.plot(
            [xmin, xmax], [0, 0], color="k", linewidth=self.standard_thinline * 0.5
        )
        ax_plot.plot(
            [0, 0], [ymin, ymax], color="k", linewidth=self.standard_thinline * 0.5
        )

        self.save_figure(fig, savename)
