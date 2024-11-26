import matplotlib.pyplot as plt


def font_setting():
    plt.rcParams["figure.dpi"] = 350            # dpi(dots per inch)
    # Whether to use automatic layout adjustment
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.subplot.left"] = 0.14  # Blank
    plt.rcParams["figure.subplot.bottom"] = 0.14  # Blank
    plt.rcParams["figure.subplot.right"] = 0.90  # Blank
    plt.rcParams["figure.subplot.top"] = 0.90   # Blank
    # Horizontal spacing of figure
    plt.rcParams["figure.subplot.wspace"] = 0.20
    plt.rcParams["figure.subplot.hspace"] = 0.20  # Vertical spacing of figure

    plt.rcParams["font.family"] = "Times New Roman"       # Font
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["font.size"] = 16              # Basic size of font
    plt.rcParams["mathtext.cal"] = "serif"      # Font for TeX
    plt.rcParams["mathtext.rm"] = "serif"       # Font for TeX
    plt.rcParams["mathtext.it"] = "serif:italic"  # Font for TeX
    plt.rcParams["mathtext.bf"] = "serif:bold"  # Font for TeX
    plt.rcParams["mathtext.fontset"] = "cm"     # Font for TeX

    # Scale line orientation, inner "in" or outer "out" or both "inout"
    plt.rcParams["xtick.direction"] = "in"
    # Scale line orientation, inner "in" or outer "out" or both "inout"
    plt.rcParams["ytick.direction"] = "in"
    # Whether to draw a scale line at the top
    plt.rcParams["xtick.top"] = True
    # Whether to draw a scale line at the bottom
    plt.rcParams["xtick.bottom"] = True
    # Whether to draw a scale line at the left
    plt.rcParams["ytick.left"] = True
    # Whether to draw a scale line at the right
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.major.size"] = 2.0      # x-axis main scale line length
    plt.rcParams["ytick.major.size"] = 2.0      # y-axis main scale line length
    plt.rcParams["xtick.major.width"] = 0.5     # x-axis main scale line width
    plt.rcParams["ytick.major.width"] = 0.5     # y-axis main scale line width
    # whether to draw x-axis minor tick marks
    plt.rcParams["xtick.minor.visible"] = False
    # whether to draw y-axis minor tick marks
    plt.rcParams["ytick.minor.visible"] = False
    # x-axis minor scale line length
    plt.rcParams["xtick.minor.size"] = 1.0
    # y-axis minor scale line length
    plt.rcParams["ytick.minor.size"] = 1.0
    plt.rcParams["xtick.minor.width"] = 0.3     # x-axis minor scale line width
    plt.rcParams["ytick.minor.width"] = 0.3     # y-axis minor scale line width
    plt.rcParams["xtick.labelsize"] = 16         # Font size of graph x-scale
    plt.rcParams["ytick.labelsize"] = 16        # Font size of graph x-scale

    plt.rcParams["axes.labelsize"] = 16         # Axis label font size
    # Line thickness around the graph
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["axes.grid"] = True            # Whether to show the grid

    plt.rcParams["grid.color"] = "black"        # Grid color
    plt.rcParams["grid.linewidth"] = 0.05       # Grid width

    plt.rcParams["legend.loc"] = "best"         # Legend position
    # Whether to surround the legend
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 1.0     # Transmittance alpha
    plt.rcParams["legend.facecolor"] = "white"  # Background color
    plt.rcParams["legend.edgecolor"] = "black"  # Enclosure color
    # Set to True to round the four corners of the enclosure
    plt.rcParams["legend.fancybox"] = False