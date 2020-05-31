"""Filled with junk code from grad school days.

Not organized or cleaned at all. Would ignore, but feel free to use anything
you find useful here
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.stats as scistat

# from scipy import ndimage
import os as os
import math

from matplotlib.colors import LinearSegmentedColormap


"""
cdict_custom = {'red':  ((0.0, 1.0, 1.0),
                   (0.25, 0.5, 0.5),
                   (0.5, 0.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.75, 0.5, 0.5),
                   (1.0, 1.0, 1.0)),
         'blue':  ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'alpha': ((0.0, 1.0, 1.0),
                   (0.25, 1.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 1.0, 1.0))
        }
"""

cdict_custom = {
    "red": (
        (0.0, 0.75, 0.75),
        (0.25, 0.825, 0.825),
        (0.5, 1.0, 1.0),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    "green": (
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (0.75, 0.825, 0.825),
        (1.0, 0.75, 0.75),
    ),
    "blue": (
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    #         'alpha': ((0.0, 1.0, 1.0),
    #                   (0.4, 1.0, 1.0),
    #                   (0.5, 0.0, 0.0),
    #                   (0.6, 1.0, 1.0),
    #                   (1.0, 1.0, 1.0))
}

CUSTOM_RWG = LinearSegmentedColormap("CUSTOM_RWG", cdict_custom)

"""
Define All Functions Here
"""


def check_quadrants(avalue, bvalue):
    total = 0
    correct = 0
    for a, b in zip(avalue, bvalue):
        if a > 0 and b > 0:
            correct += 1
        elif a < 0 and b < 0:
            correct += 1
        total += 1

    return correct, total


def ensure_dir(this_dir):
    if this_dir[0] == "/":
        test_dir = "/"
    else:
        test_dir = ""
    all_dir = this_dir.strip().split("/")
    for thing in all_dir:
        test_dir += thing
        if os.path.isdir(test_dir):
            pass
        else:
            os.mkdir(test_dir)
        test_dir += "/"


def get_log_int(val):
    return np.log10(val).astype(int)


def floor_logarithm(val):
    new_ints = get_log_int(val)
    return np.array([10.0 ** val for val in new_ints])


def get_color_lists():
    from matplotlib import colors as mcolors

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    by_hsv = sorted(
        (tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
        for name, color in colors.items()
    )
    sorted_names = [name for hsv, name in by_hsv]
    return sorted_names


def load_contact_files(source_dir, load_order):
    contacts_list = []
    probability_list = []
    for i in load_order:
        temp_list = read_pairs_file("%s/state_%d_pairs.dat" % (source_dir, i))
        contacts_list.append(temp_list)
        temp_prob_list = read_probability_file(
            "%s/state_%d_probability.dat" % (source_dir, i)
        )
        probability_list.append(temp_prob_list)

    return contacts_list, probability_list


def read_pairs_file(file_name):
    temp_list = []
    f = open(file_name, "r")
    for line in f:
        stuff = line.strip().split()
        assert len(stuff) == 2
        temp_list.append([int(stuff[0]), int(stuff[1])])
    return temp_list


def read_probability_file(file_name):
    temp_list = []
    f = open(file_name)
    for line in f:
        stuff = line.strip().split()
        assert len(stuff) == 1
        temp_list.append(float(stuff[0]))
    return temp_list


def get_labels(labels, order):
    assert len(labels) == len(order)
    correct_list = [None] * len(labels)
    for count, idx in enumerate(order):
        correct_list[idx] = labels[count]
    for val in correct_list:
        assert val is not None
    return correct_list


def get_general_map(
    top_list, top_probability, bottom_list, bottom_probability, n_residues=66
):

    contact_matrix = np.zeros((n_residues, n_residues))
    native_count = 0
    for index in range(np.shape(top_list)[0]):
        idx = top_list[index, 0]
        jdx = top_list[index, 1]
        contact_matrix[jdx, idx] = top_probability[index]
    for index in range(np.shape(bottom_list)[0]):
        idx = bottom_list[index, 0]
        jdx = bottom_list[index, 1]
        contact_matrix[idx, jdx] = bottom_probability[index]
    edges = np.arange(0.5, n_residues + 1, 1)
    return contact_matrix, edges


def get_contact_map(
    contact_list,
    contact_probability,
    native_contact_list,
    native_contact_probability,
    native_cutoff=0.8,
    cutoff=0.3,
):
    n_residues = 35

    contact_matrix = np.zeros((n_residues, n_residues))
    native_count = 0
    for pair_idx, pair in enumerate(native_contact_list):
        if native_contact_probability[pair_idx] >= native_cutoff:
            contact_matrix[pair[0], pair[1]] = 1
            native_count += 1
    for pair_idx, pair in enumerate(contact_list):
        p0 = int(pair[0])
        p1 = int(pair[1])
        if contact_probability[pair_idx] > cutoff:
            if contact_matrix[p0, p1] == 1:
                contact_matrix[p1, p0] = contact_probability[pair_idx]
            else:
                contact_matrix[p1, p0] = -contact_probability[pair_idx]

    # print "Number of Native Contacts = %d" % native_count
    return contact_matrix


def _func_gauss(x, mu, sigma, total=1.0):
    if sigma == 0:
        sigma = 10 ** -16
    scale = total / math.sqrt(2.0 * math.pi * (sigma ** 2))
    exponent = -1.0 * ((x - mu) ** 2) / (2.0 * (sigma ** 2))

    results = scale * np.exp(exponent)

    return results


def compute_gaussian_and_chi(decoyE, spacing=0.2):

    n_decoys = float(np.shape(decoyE)[0])
    if n_decoys > 0:
        max_value = ((math.ceil(np.max(np.abs(decoyE)) / spacing)) * spacing) + spacing
        ebins = np.arange(-max_value - (0.5 * spacing), max_value + spacing, spacing)
        avg = np.sum(decoyE) / n_decoys
        sd = np.sqrt(np.sum((decoyE - avg) ** 2) / n_decoys)
        hist_values, bin_edges = np.histogram(decoyE, bins=ebins)
        sd_values = np.sqrt(hist_values.astype(float))
        center_values = (bin_edges[1:] + bin_edges[:-1]) * 0.5
        true_values = _func_gauss(center_values, avg, sd, total=n_decoys * spacing)
        chi_pieces = ((hist_values - true_values) ** 2) / (sd_values ** 2)
        chi_pieces[np.where(hist_values == 0)] = 0
        chi = np.sum(chi_pieces) / float(np.shape(center_values)[0])
    else:
        chi = 0
        avg = 0
        sd = 1

    return chi, avg, sd, n_decoys * spacing, hist_values, bin_edges


class Plotter(object):
    def __init__(self, cwd=None, save_dir_name="test"):
        if cwd is None:
            self.cwd = os.getcwd()
        else:
            self.cwd = cwd

        self.set_save_dir(save_dir_name)

        self.colors = ["b", "r", "g", "m", "c", "y"]
        self.colors_fep = ["b", "r", "g", "m", "c"]
        self.colors_black = ["k"]

        self.line_types = ["-", "--", ":", "-."]

        self.alphabet = [chr(i) for i in range(ord("a"), ord("z") + 1)]

        for col in self.colors:
            self.colors_black.append(col)

    def set_save_dir(self, save_dir_name):
        self.figures_dir = "%s/%s" % (
            self.cwd,
            save_dir_name,
        )  # directory for saving final figures
        self.png_figures_dir = "%s/%s_png" % (self.cwd, save_dir_name)

        ensure_dir(self.figures_dir)
        ensure_dir(self.png_figures_dir)

    def save_figure(self, fig, savefile):
        fig.savefig(
            "%s/%s.pdf" % (self.figures_dir, savefile),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=300,
        )

        png_save_name = "%s/%s" % (self.png_figures_dir, savefile)

        fig.savefig(
            "%s.png" % png_save_name, bbox_inches="tight", format="png", dpi=300
        )
        # fig.savefig("%s.png" % png_save_name, format="png", dpi=300)

    def set_presentation_dimensions(self):
        self.column_width_inches = 5.5  # width of column in a two-column article
        self.double_column_width_inches = 7.0  # width of a two-column spanning figure
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

        matplotlib.rcParams.update({"font.size": 18})
        matplotlib.rcParams.update({"axes.labelsize": 18})

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

        matplotlib.rcParams.update({"font.size": 9})
        matplotlib.rcParams.update({"axes.labelsize": 9})

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

        matplotlib.rcParams.update({"font.size": 9})
        matplotlib.rcParams.update({"axes.labelsize": 9})

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

    def plot_spread(self, data, savename, bins=None, xname="data", yname="count"):
        if bins is None:
            bins = np.sqrt(len(data))
            bins = int(bins)
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

        ax_plot.hist(data, bins=bins)

        ax_plot.set_xlabel(xname)
        ax_plot.set_ylabel(yname)

        self.save_figure(fig, savename)

    def plot_many_images(self, image_list, savename, ncols=1, nrows=1):
        first_im = mpimg.imread("%s" % image_list[0])
        ratio = float(np.shape(first_im)[0]) / float(np.shape(first_im)[1])
        assert len(image_list) == ncols * nrows

        fig_width_inches = self.column_width_inches
        fig_height_padding_inches = self.padding_standard_width_inches

        image_height_inches = ratio * fig_width_inches / ncols

        fig_height_inches = (fig_height_padding_inches + image_height_inches) * nrows
        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))

        image_height = image_height_inches / fig_height_inches
        vertical_padding = fig_height_padding_inches / fig_height_inches
        vertical_diff = vertical_padding + image_height
        highest_vertical_start = 1.0 - vertical_diff
        for count, image in enumerate(image_list):
            horizontal_start = (1.0 / (ncols)) * (count % 2)
            vertical_start = highest_vertical_start - (
                vertical_diff * (np.floor(count / ncols))
            )
            ax1 = fig.add_axes(
                [horizontal_start, vertical_start, 1.0 / ncols, image_height]
            )
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            ax1.axis("off")
            this_img = mpimg.imread("%s" % image_list[count])
            ax1.imshow(this_img)
            ax1.set_title(
                "(%s)" % self.alphabet[count], fontsize=self.standard_special_font_size
            )

        self.save_figure(fig, savename)

    def plot_two_images(self, image1, image2, savename, labels1=[], labels2=[]):
        first_im = mpimg.imread("%s" % image1)
        second_im = mpimg.imread("%s" % image2)
        ratio = float(np.shape(first_im)[0]) / float(np.shape(first_im)[1])
        fig_width_inches = self.column_width_inches
        fig_height_padding = self.padding_standard_width_inches
        image_height_inches = ratio * fig_width_inches / 2.0

        fig_height_inches = fig_height_padding + image_height_inches
        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))

        ax_height = image_height_inches / fig_height_inches
        ax1 = fig.add_axes([0.0, 0.0, 0.5, ax_height])
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        ax1.axis("off")

        ax2 = fig.add_axes([0.5, 0.0, 0.5, ax_height])
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.axis("off")

        this_image = ax1.imshow(first_im)
        ax1.set_title("(a)", fontsize=self.standard_special_font_size)
        for thing in labels1:
            ax1.text(thing[0], thing[1], thing[2], fontsize=self.standard_font_size)
        this_image = ax2.imshow(second_im)
        ax2.set_title("(b)", fontsize=self.standard_special_font_size)
        for thing in labels2:
            ax2.text(thing[0], thing[1], thing[2], fontsize=self.standard_font_size)

        self.save_figure(fig, savename)

    def plot_scatter(self, data, savename, xname="data", yname="count", axis=None):
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

        ax_plot.scatter(data[:, 0], data[:, 1])

        ax_plot.set_xlabel(xname)
        ax_plot.set_ylabel(yname)
        if axis is not None:
            ax_plot.axis(axis)
        else:
            xmax = np.max(np.abs(data[:, 0])) * 1.05
            ymax = np.max(np.abs(data[:, 1])) * 1.05
            ax_plot.axis([-xmax, xmax, -ymax, ymax])

        ymin, ymax = ax_plot.get_ylim()
        xmin, xmax = ax_plot.get_xlim()
        ax_plot.plot(
            [xmin, xmax], [0, 0], color="k", linewidth=self.standard_thinline * 0.5
        )
        ax_plot.plot(
            [0, 0], [ymin, ymax], color="k", linewidth=self.standard_thinline * 0.5
        )

        self.save_figure(fig, savename)

        # compute r2 value
        r, p = scistat.pearsonr(data[:, 0], data[:, 1])
        correct, total = check_quadrants(data[:, 0], data[:, 1])

        return r ** 2, correct, total

    def plot_multilines(
        self,
        list_of_x,
        list_of_y,
        list_of_labels,
        savename,
        use_colors=None,
        mark_values=None,
    ):
        n_things = np.shape(list_of_x)[0]
        if use_colors is None:
            use_colors = self.colors
        fig_width_inches = self.column_width_inches
        fig_height_inches = self.standard_box_height_inches

        fig_ratio = fig_height_inches / fig_width_inches
        figure_size = (fig_width_inches, fig_height_inches)
        print "fig_width = %f, fig_ratio = %f" % (fig_width_inches, fig_ratio)
        fig = plt.figure(figsize=figure_size)

        left_padding_inches = self.padding_standard_width_inches + 0.025
        right_padding_inches = self.padding_empty_inches
        top_padding_inches = self.padding_empty_inches
        bottom_padding_inches = self.padding_standard_height_inches + 0.05

        axes_width_inches = (
            fig_width_inches - left_padding_inches - right_padding_inches
        )
        axes_height_inches = (
            fig_height_inches - top_padding_inches - bottom_padding_inches
        )

        print "axes_ratio = %f" % (axes_height_inches / axes_width_inches)

        left_padding = left_padding_inches / fig_width_inches
        right_padding = right_padding_inches / fig_width_inches
        top_padding = top_padding_inches / fig_height_inches
        bottom_padding = bottom_padding_inches / fig_height_inches

        axes_width = axes_width_inches / fig_width_inches
        axes_height = axes_height_inches / fig_height_inches
        print "stuff = %f %f %f %f" % (
            left_padding,
            axes_width,
            bottom_padding,
            axes_height,
        )
        this_axes = [left_padding, bottom_padding, axes_width, axes_height]

        ax_plot = fig.add_axes(this_axes)
        ax_plot.set_xlabel("Residue Index")
        ax_plot.set_ylabel("Z-Score")
        # ax_plot.add_legend()
        max_x = 0
        max_y = 0
        for idx in range(n_things):
            this_x = list_of_x[idx]
            this_y = list_of_y[idx]
            print "### test ###"
            print np.shape(this_x)
            print np.shape(this_y)
            ax_plot.plot(
                list_of_x[idx],
                list_of_y[idx],
                label=list_of_labels[idx],
                color=use_colors[idx],
                linewidth=1,
                alpha=0.75,
            )

            new_x = np.max(this_x)
            new_y = np.max(np.abs(this_y))
            if max_x < new_x:
                max_x = new_x
            if max_y < new_y:
                max_y = new_y

        ax_plot.axis([0, max_x, -max_y * 1.1, max_y * 1.1])

        ax_plot.plot([0, max_x], [0, 0], color="k", linewidth=0.5, linestyle="--")
        for val in mark_values:
            ax_plot.plot(
                [val, val],
                [-max_y * 1.1, max_y * 1.1],
                linewidth=0.5,
                linestyle="--",
                color="k",
            )
        plt.legend()
        self.save_figure(fig, savename)

    def plot_two_histograms(self, list_of_data, savename):
        # fig_width_inches = self.double_column_width_inches
        fig_width_inches = 6
        fig_height_inches = self.standard_box_height_inches

        fig_ratio = fig_height_inches / fig_width_inches
        figure_size = (fig_width_inches, fig_height_inches)
        print "fig_width = %f, fig_ratio = %f" % (fig_width_inches, fig_ratio)
        fig = plt.figure(figsize=figure_size)

        left_padding_inches = self.padding_standard_width_inches
        right_padding_inches = self.padding_empty_inches
        top_padding_inches = self.padding_empty_inches
        bottom_padding_inches = self.padding_standard_height_inches

        axes_width_inches = (
            fig_width_inches - 2 * left_padding_inches - 2 * right_padding_inches
        ) / 2
        axes_height_inches = (
            fig_height_inches - top_padding_inches - bottom_padding_inches
        )

        print "axes_ratio = %f" % (axes_height_inches / axes_width_inches)

        left_padding = left_padding_inches / fig_width_inches
        right_padding = right_padding_inches / fig_width_inches
        top_padding = top_padding_inches / fig_height_inches
        bottom_padding = bottom_padding_inches / fig_height_inches

        axes_width = axes_width_inches / fig_width_inches
        axes_height = axes_height_inches / fig_height_inches
        print "stuff = %f %f %f %f" % (
            left_padding,
            axes_width,
            bottom_padding,
            axes_height,
        )
        first_axes = [left_padding, bottom_padding, axes_width, axes_height]
        second_axes = [left_padding + 0.5, bottom_padding, axes_width, axes_height]
        ax_first = fig.add_axes(first_axes)
        ax_second = fig.add_axes(second_axes)
        self._plot_histogram_line(ax_first, list_of_data[0], label="(a)")
        self._plot_histogram_line(ax_second, list_of_data[1], label="(b)")

        self.save_figure(fig, savename)

    def _plot_histogram_line(self, ax_main, e_data, label=None):
        chi, avg, sd, scale, hist_values, bin_edges = compute_gaussian_and_chi(e_data)
        print "Chi value is: %f" % chi
        print "log10 Chi Value is: %f" % np.log10(chi)
        ax_main.hist(e_data, bins=bin_edges, color="b", edgecolor="b")
        start = bin_edges[0]
        end = bin_edges[-1]
        x_values = np.arange(start, end + 0.01, 0.01)
        y_values = _func_gauss(x_values, avg, sd, total=scale)
        ax_main.plot(x_values, y_values, color="r")

        ax_main.set_xlabel("E")
        ax_main.set_ylabel("Count")

        if label is not None:
            ax_main.set_title(label)

    def _plot_map_on_axes(
        self,
        ax_main,
        edges,
        plot_map,
        this_cmap,
        cmap_low,
        cmap_high,
        dca_pairs=None,
        mark_values=None,
        native_pairs=None,
        grid_color="green",
        dca_color="r",
    ):
        n_residues = np.shape(plot_map)[0]

        grid_width = 0.5
        cross_width = 0.5
        x_shift = 0.0
        y_shift = 0.0
        alpha = 1.0

        if native_pairs is not None:
            for index in range(np.shape(native_pairs)[0]):
                idx = native_pairs[index, 0]
                jdx = native_pairs[index, 1]
                left = edges[idx] - x_shift
                right = edges[idx + 1] - x_shift
                bottom = edges[jdx] - y_shift
                top = edges[jdx + 1] - y_shift
                ax_main.plot(
                    [left, left],
                    [bottom, top],
                    linewidth=grid_width,
                    color=grid_color,
                    alpha=alpha,
                )
                ax_main.plot(
                    [right, right],
                    [bottom, top],
                    linewidth=grid_width,
                    color=grid_color,
                    alpha=alpha,
                )
                ax_main.plot(
                    [left, right],
                    [top, top],
                    linewidth=grid_width,
                    color=grid_color,
                    alpha=alpha,
                )
                ax_main.plot(
                    [left, right],
                    [bottom, bottom],
                    linewidth=grid_width,
                    color=grid_color,
                    alpha=alpha,
                )

                # other side
                left = edges[jdx] - x_shift
                right = edges[jdx + 1] - x_shift
                bottom = edges[idx] - y_shift
                top = edges[idx + 1] - y_shift
                ax_main.plot(
                    [left, left],
                    [bottom, top],
                    linewidth=grid_width,
                    color=grid_color,
                    alpha=alpha,
                )
                ax_main.plot(
                    [right, right],
                    [bottom, top],
                    linewidth=grid_width,
                    color=grid_color,
                    alpha=alpha,
                )
                ax_main.plot(
                    [left, right],
                    [top, top],
                    linewidth=grid_width,
                    color=grid_color,
                    alpha=alpha,
                )
                ax_main.plot(
                    [left, right],
                    [bottom, bottom],
                    linewidth=grid_width,
                    color=grid_color,
                    alpha=alpha,
                )

        # dca pairs
        if dca_pairs is not None and show_this_dca[count]:
            for index in range(np.shape(dca_pairs)[0]):
                idx = dca_pairs[index, 0]
                jdx = dca_pairs[index, 1]
                left = edges[idx] - x_shift
                right = edges[idx + 1] - x_shift
                bottom = edges[jdx] - y_shift
                top = edges[jdx + 1] - y_shift

                ax_main.plot(
                    [left, right],
                    [bottom, top],
                    linewidth=cross_width,
                    color=dca_color,
                    alpha=alpha,
                )
                ax_main.plot(
                    [left, right],
                    [top, bottom],
                    linewidth=cross_width,
                    color=dca_color,
                    alpha=alpha,
                )

                # other side
                left = edges[jdx] - x_shift
                right = edges[jdx + 1] - x_shift
                bottom = edges[idx] - y_shift
                top = edges[idx + 1] - y_shift
                if not top_half_only:
                    ax_main.plot(
                        [left, right],
                        [bottom, top],
                        linewidth=cross_width,
                        color=dca_color,
                        alpha=alpha,
                    )
                    ax_main.plot(
                        [left, right],
                        [top, bottom],
                        linewidth=cross_width,
                        color=dca_color,
                        alpha=alpha,
                    )

        if mark_values is not None:
            for m_value in mark_values:
                # horizontal lines
                ax_main.plot(
                    [0, n_residues + 1],
                    [m_value, m_value],
                    linewidth=cross_width,
                    color=dca_color,
                    alpha=alpha,
                    linestyle=":",
                )

                # vertical lines
                ax_main.plot(
                    [m_value, m_value],
                    [0, n_residues + 1],
                    linewidth=cross_width,
                    color=dca_color,
                    alpha=alpha,
                    linestyle=":",
                )

        ax_main.grid(
            "on",
            which="minor",
            axis="both",
            linestyle="-",
            color="gray",
            linewidth=grid_width,
        )
        tick_range = np.arange(0, n_residues + 1, 5)
        ax_main.set_xticks(tick_range, minor=True)
        ax_main.set_yticks(tick_range, minor=True)

        new_map = np.copy(plot_map)
        new_map = np.ma.masked_where(new_map == 0, new_map)
        # qmesh = ax_main.pcolormesh(edges, edges, plot_map, vmin=-maxval, vmax=maxval, cmap="bwr_r")
        qmesh = ax_main.pcolormesh(
            edges, edges, new_map, vmin=cmap_low, vmax=cmap_high, cmap=this_cmap
        )
        ax_main.plot([0, n_residues + 1], [0, n_residues + 1], color="k", linewidth=1)
        ax_main.axis([0, n_residues + 1, 0, n_residues + 1])

        return qmesh

    def plot_epsilon_map_with_images(
        self,
        map,
        edges,
        image1,
        image2,
        savename,
        double_column=False,
        color_map_bounds=[-3, 3],
    ):

        first_im = mpimg.imread("%s" % image1)
        second_im = mpimg.imread("%s" % image2)
        ratio = float(np.shape(first_im)[0]) / float(np.shape(first_im)[1])

        if double_column:
            fig_width_inches = self.double_column_width_inches
        else:
            fig_width_inches = self.column_width_inches
        left_padding_inches = self.padding_standard_width_inches
        right_padding_inches = self.padding_empty_inches
        inter_column_padding_inches = self.padding_buffer_inches * 2.0
        inter_row_padding_inches = self.padding_standard_height_inches * 1.5

        square_size = fig_width_inches - left_padding_inches - right_padding_inches

        color_bar_height = self.padding_buffer_inches
        color_bar_padding_inches = (
            2 * self.padding_buffer_inches + self.padding_standard_height_inches
        )

        image_width_padding_inches = 0.05

        image_height_inches = (
            ratio * (fig_width_inches - image_width_padding_inches) / 2.0
        )

        fig_height_inches = (
            square_size
            + (
                (2 * self.padding_standard_height_inches)
                + self.padding_title_height_inches
            )
            + color_bar_height
            + color_bar_padding_inches
        )  # the one epsilon map
        fig_height_inches += (
            self.padding_standard_height_inches + image_height_inches
        )  # images

        fig_ratio = fig_height_inches / fig_width_inches

        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))

        # generate positions of axes for epsilon map:
        inter_column_padding = inter_column_padding_inches / fig_width_inches
        padding_vertical = self.padding_standard_height_inches / fig_height_inches
        inter_row_padding = inter_row_padding_inches / fig_height_inches

        left_padding = left_padding_inches / fig_width_inches
        right_padding = left_padding

        max_width = 1.0 - left_padding - right_padding
        color_bar_height_ratio = color_bar_height / (fig_height_inches)
        color_bar_padding = color_bar_padding_inches / fig_height_inches

        color_bar_start = padding_vertical
        vertical_usable = (square_size) / fig_height_inches

        contact_map_height = vertical_usable
        contact_map_width = contact_map_height * fig_ratio

        color_bar_width = 0.75 * max_width
        color_bar_horizontal_start = (1 - color_bar_width) / 2

        color_bar_position = [
            color_bar_horizontal_start,
            color_bar_start,
            color_bar_width,
            color_bar_height_ratio,
        ]
        contact_positions = []

        horizontal_start_position = left_padding
        vertical_start_position = (
            padding_vertical + color_bar_height_ratio + color_bar_padding
        )
        padding_title = self.padding_title_height_inches / fig_height_inches

        eps_axes = fig.add_axes(
            [
                horizontal_start_position,
                vertical_start_position,
                contact_map_width,
                contact_map_height,
            ]
        )

        this_cmap = CUSTOM_RWG
        this_cmap.set_bad(color="white")

        cmap_low = color_map_bounds[0]
        cmap_high = color_map_bounds[1]

        qmesh = self._plot_map_on_axes(
            eps_axes, edges, map, this_cmap, cmap_low, cmap_high
        )
        eps_axes.set_xlabel("i")
        eps_axes.set_ylabel("j")
        eps_axes.set_title("(c)")

        ax_cb = fig.add_axes(color_bar_position)
        ax_cb.xaxis.set_ticks_position("top")
        cb = fig.colorbar(
            qmesh, cax=ax_cb, ticks=[cmap_low, -0, cmap_high], orientation="horizontal"
        )
        ax_cb.set_xlabel("${F}_{ij}$", labelpad=-24)

        # now for the images:
        image_start_position = (
            vertical_start_position + contact_map_height + padding_title
        )
        image_width_padding = image_width_padding_inches / fig_width_inches
        image_width = (1.0 - image_width_padding) / 2.0

        im1_horizontal_start = 0
        im2_horizontal_start = 0.5 + (0.5 * image_width_padding)

        ax_height = image_height_inches / fig_height_inches
        ax1 = fig.add_axes([0.0, image_start_position, image_width, ax_height])
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        ax1.axis("off")

        ax2 = fig.add_axes(
            [im2_horizontal_start, image_start_position, image_width, ax_height]
        )
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.axis("off")

        ax1.imshow(first_im)
        ax1.set_title("(a)", fontsize=self.standard_special_font_size)
        ax2.imshow(second_im)
        ax2.set_title("(b)", fontsize=self.standard_special_font_size)

        self.save_figure(fig, savename)

    def plot_epsilon_map(
        self,
        list_of_plot_map,
        list_of_edges,
        savename,
        native_pairs=None,
        dca_pairs=None,
        mark_values=None,
        ncols=1,
        nrows=2,
        show_colorbar=True,
        label_columns=True,
        show_this_dca=None,
        top_half_only=True,
        grid_color="green",
        dca_color="r",
        use_color_map=None,
        color_map_bounds=[-3, 3],
        double_column=False,
        colorbar_label="${F}_{ij}$",
    ):
        cmap_low = color_map_bounds[0]
        cmap_high = color_map_bounds[1]

        # fig_ratio = 1.
        if double_column:
            fig_width = self.double_column_width_inches
        else:
            fig_width = self.column_width_inches
        left_padding_inches = self.padding_standard_width_inches
        right_padding_inches = self.padding_empty_inches
        inter_column_padding_inches = self.padding_buffer_inches * 2.0
        inter_row_padding_inches = self.padding_standard_height_inches * 1.5
        square_size = (
            fig_width
            - left_padding_inches
            - right_padding_inches
            - ((ncols - 1) * inter_column_padding_inches)
        ) / ncols

        color_bar_height = self.padding_buffer_inches
        color_bar_padding_inches = (
            2 * self.padding_buffer_inches + self.padding_standard_height_inches
        )

        fig_height = (
            nrows * (square_size)
            + ((nrows - 1) * inter_row_padding_inches)
            + (
                (2 * self.padding_standard_height_inches)
                + self.padding_title_height_inches
            )
            + color_bar_height
            + color_bar_padding_inches
        )

        fig_ratio = fig_height / fig_width
        figure_size = (fig_width, fig_height)
        fig = plt.figure(figsize=figure_size)

        n_contact_maps = len(list_of_plot_map)
        try:
            assert n_contact_maps == ncols * nrows
        except:
            print n_contact_maps
            print ncols * nrows
            raise

        # generate positions of each axes:
        inter_column_padding = inter_column_padding_inches / fig_width
        padding_vertical = self.padding_standard_height_inches / fig_height
        inter_row_padding = inter_row_padding_inches / fig_height

        left_padding = left_padding_inches / fig_width
        right_padding = left_padding

        max_width = 1.0 - left_padding - right_padding
        color_bar_height_ratio = color_bar_height / (fig_height)
        color_bar_padding = color_bar_padding_inches / fig_height

        color_bar_start = padding_vertical
        vertical_usable = (square_size * nrows) / fig_height

        contact_map_height = vertical_usable / nrows
        contact_map_width = contact_map_height * fig_ratio

        color_bar_width = 0.75 * max_width
        color_bar_horizontal_start = (1 - color_bar_width) / 2

        color_bar_position = [
            color_bar_horizontal_start,
            color_bar_start,
            color_bar_width,
            color_bar_height_ratio,
        ]
        contact_positions = []

        horizontal_start_position = left_padding
        vertical_start_position = (
            padding_vertical + color_bar_height_ratio + color_bar_padding
        )

        for i in range(nrows):
            for j in range(ncols):
                this_h_start = horizontal_start_position + (
                    (contact_map_width + inter_column_padding) * j
                )
                this_v_start = vertical_start_position + (
                    (contact_map_height + inter_row_padding) * (nrows - 1 - i)
                )
                contact_positions.append(
                    [this_h_start, this_v_start, contact_map_width, contact_map_height]
                )

        if show_this_dca is None:
            show_this_dca = [True for i in range(ncols * nrows)]

        # first draw a line around every native pair
        grid_width = 0.25
        cross_width = 0.5
        alpha = 1.0
        x_shift = 0.0
        y_shift = 0.03
        count = 0
        if use_color_map is None:
            this_cmap = CUSTOM_RWG
            # this_cmap = plt.cm.bwr_r
        elif use_color_map == "bwr_r":
            this_cmap = plt.cm.bwr_r
        elif use_color_map == "RdYlGn":
            this_cmap = plt.cm.RdYlGn
        elif use_color_map == "hot":
            this_cmap = plt.cm.hot
        elif use_color_map == "viridis":
            this_cmap = plt.cm.viridis
        else:
            print "No Map Selected"
            this_cmap = plt.cm.bwr_r
        this_cmap.set_bad(color="white")

        for i in range(nrows):
            for j in range(ncols):
                ax_main = fig.add_axes(contact_positions[count])
                edges = list_of_edges[count]
                plot_map = list_of_plot_map[count]
                qmesh = self._plot_map_on_axes(
                    ax_main,
                    edges,
                    plot_map,
                    this_cmap,
                    cmap_low,
                    cmap_high,
                    native_pairs=native_pairs,
                    dca_pairs=dca_pairs,
                    mark_values=mark_values,
                )

                dimension = ax_main.axis()
                xdiff = dimension[1] - dimension[0]
                ydiff = dimension[3] - dimension[2]
                xpos = dimension[0] + (xdiff * 0.005)
                ypos = dimension[2] + (ydiff * 0.935)
                if label_columns:
                    ax_main.set_title(
                        "(%s)" % self.alphabet[count],
                        y=1.0,
                        fontsize=self.standard_special_font_size,
                    )
                if i == (nrows - 1):
                    ax_main.set_xlabel("i")
                else:
                    pass
                    # plt.setp([ax_main.get_xticklabels()], visible=False)

                if j == 0:
                    ax_main.set_ylabel("j")
                else:
                    pass
                    # plt.setp([ax_main.get_yticklabels()], visible=False)
                count += 1

        if show_colorbar:
            ax_cb = fig.add_axes(color_bar_position)
            ax_cb.xaxis.set_ticks_position("top")
            cb = fig.colorbar(
                qmesh,
                cax=ax_cb,
                ticks=[cmap_low, -0, cmap_high],
                orientation="horizontal",
            )
            ax_cb.set_xlabel(colorbar_label, labelpad=-24)

        self.save_figure(fig, savename)
