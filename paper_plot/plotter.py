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

        self.alphabet = [chr(i) for i in range(ord('a'),ord('z')+1)]

        for col in self.colors:
            self.colors_black.append(col)

    def set_save_dir(self, save_dir_name):
        self.figures_dir = "%s/%s" % (self.cwd, save_dir_name) # directory for saving final figures
        self.png_figures_dir = "%s/%s_png" % (self.cwd, save_dir_name)

        ensure_dir(self.figures_dir)
        ensure_dir(self.png_figures_dir)

    def save_figure(self, fig, savefile):
        fig.savefig("%s/%s.pdf" % (self.figures_dir, savefile),  format="pdf", bbox_inches="tight", pad_inches=0.02, dpi=300)

        png_save_name = "%s/%s" % (self.png_figures_dir, savefile)

        fig.savefig("%s.png" % png_save_name, bbox_inches="tight", format="png", dpi=300)
        #fig.savefig("%s.png" % png_save_name, format="png", dpi=300)

    def set_presentation_dimensions(self):
        self.column_width_inches = 5.5 #width of column in a two-column article
        self.double_column_width_inches = 7. # width of a two-column spanning figure
        self.standard_box_height_inches = self.column_width_inches * (5. / 6.)
        self.padding_standard = 0.6
        self.padding_standard_width_inches = 0.5 # padding on the left for y-axis label
        self.padding_standard_height_inches = 0.5 # padding on the bottom for x-axis label
        self.padding_title_height_inches = 0.40 # padding on top for title
        self.padding_empty_inches = 0.16 # padding if no labels or tick marks exist
        self.padding_buffer_inches = 0.1 # extra padding between adjacent figures
        self.padding_buffer_xaxis = 0.3
        self.padding_buffer_yaxis = 0.3

        self.standard_font_size = 18
        self.standard_special_font_size = 27
        self.standard_thinline = 2
        self.standard_thickline = 4


        matplotlib.rcParams.update({"font.size":18})
        matplotlib.rcParams.update({"axes.labelsize":18})

        matplotlib.rcParams.update({"axes.labelpad":4.0})
        matplotlib.rcParams.update({"legend.fontsize":16})

        matplotlib.rcParams.update({"xtick.labelsize":12})
        matplotlib.rcParams.update({"xtick.major.size":3.5})
        matplotlib.rcParams.update({"xtick.minor.size":2.})
        matplotlib.rcParams.update({"xtick.major.pad":3.5})

        matplotlib.rcParams.update({"ytick.labelsize":12})
        matplotlib.rcParams.update({"ytick.major.size":3.5})
        matplotlib.rcParams.update({"ytick.minor.size":2.})
        matplotlib.rcParams.update({"ytick.major.pad":3.5})

        matplotlib.rcParams.update({"lines.linewidth":2})
        matplotlib.rcParams.update({"patch.linewidth":1})
        matplotlib.rcParams.update({"axes.linewidth":1})

    def set_acs_dimensions(self):
        self.column_width_inches = 3.25 #width of column in a two-column article
        self.double_column_width_inches = 7. # width of a two-column spanning figure
        self.standard_box_height_inches = self.column_width_inches * (5. / 6.)
        self.padding_standard = 0.4
        self.padding_standard_width_inches = 0.32 # padding on the left for y-axis label
        self.padding_standard_width_label_inches = 0.2 # padding on the left for the y-axis when it is labeled but no tick-labels
        self.padding_standard_height_inches = 0.25 # padding on the bottom for x-axis label
        self.padding_title_height_inches = 0.20 # padding on top for title
        self.padding_empty_inches = 0.08 # padding if no labels or tick marks exist
        self.padding_buffer_inches = 0.1 # extra padding between adjacent figures
        self.padding_buffer_xaxis = 0.15 # extra padding along vertical dimension
        self.padding_buffer_yaxis = 0.15 # extra padding along horizontal dimension

        self.standard_font_size = 9
        self.standard_special_font_size = 13
        self.standard_thinline = 1
        self.standard_thickline = 2

        matplotlib.rcParams.update({"font.size":9})
        matplotlib.rcParams.update({"axes.labelsize":9})

        matplotlib.rcParams.update({"axes.labelpad":2.0})
        matplotlib.rcParams.update({"legend.fontsize":8})

        matplotlib.rcParams.update({"xtick.labelsize":6})
        matplotlib.rcParams.update({"xtick.major.size":1.75})
        matplotlib.rcParams.update({"xtick.minor.size":1.})
        matplotlib.rcParams.update({"xtick.major.pad":1.75})

        matplotlib.rcParams.update({"ytick.labelsize":6})
        matplotlib.rcParams.update({"ytick.major.size":1.75})
        matplotlib.rcParams.update({"ytick.minor.size":1.})
        matplotlib.rcParams.update({"ytick.major.pad":1.75})

        matplotlib.rcParams.update({"lines.linewidth":1})
        matplotlib.rcParams.update({"patch.linewidth":0.5})
        matplotlib.rcParams.update({"axes.linewidth":0.5})