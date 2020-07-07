import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Selection of palettes for cluster coloring, and scatter values
magma = [plt.get_cmap("magma")(i) for i in np.linspace(0, 1, 100)]
magma[0] = (0.88, 0.88, 0.88, 1)
magma = mpl.colors.LinearSegmentedColormap.from_list("", magma[:80])

# Discrete palette [Combination of BOLD and VIVID from carto colors]
bold_and_vivid = [
    "#7F3C8D",
    "#11A579",
    "#3969AC",
    "#F2B701",
    "#E73F74",
    "#80BA5A",
    "#E68310",
    "#008695",
    "#CF1C90",
    "#f97b72",
    "#E58606",
    "#4b4b8f",
    "#5D69B1",
    "#52BCA3",
    "#99C945",
    "#CC61B0",
    "#24796C",
    "#DAA51B",
    "#2F8AC4",
    "#764E9F",
    "#ED645A",
    "#CC3A8E",
]

prism = [
    "#5F4690",
    "#1D6996",
    "#38A6A5",
    "#0F8554",
    "#73AF48",
    "#EDAD08",
    "#E17C05",
    "#CC503E",
    "#94346E",
    "#6F4070",
    "#994E95",
]
prism = prism[::2] + prism[1::2]
safe = [
    "#88CCEE",
    "#CC6677",
    "#DDCC77",
    "#117733",
    "#332288",
    "#AA4499",
    "#44AA99",
    "#999933",
    "#882255",
    "#661100",
    "#6699CC",
]
vivid = [
    "#E58606",
    "#5D69B1",
    "#52BCA3",
    "#99C945",
    "#CC61B0",
    "#24796C",
    "#DAA51B",
    "#2F8AC4",
    "#764E9F",
    "#ED645A",
    "#CC3A8E",
]
bold = [
    "#7F3C8D",
    "#11A579",
    "#3969AC",
    "#F2B701",
    "#E73F74",
    "#80BA5A",
    "#E68310",
    "#008695",
    "#CF1C90",
    "#f97b72",
    "#4b4b8f",
]
# Diverging palettes
temps = [
    "#009392",
    "#39b185",
    "#9ccb86",
    "#e9e29c",
    "#eeb479",
    "#e88471",
    "#cf597e",
]

# Continuous palettes
teal = [
    "#d1eeea",
    "#a8dbd9",
    "#85c4c9",
    "#68abb8",
    "#4f90a6",
    "#3b738f",
    "#2a5674",
]
