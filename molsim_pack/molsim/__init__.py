from ._molsim import HardDisks, MonteCarlo, MolecularDynamics
from .mullerBrown import (
    mullerBrownPotential,
    mullerBrownGradient_dx,
    mullerBrownGradient_dy,
    mullerBrownHeatmap,
    mullerBrownPotentialAndGradient,
    plot_muller_brown_heatmap,
)
from .utils import blockAverage

import matplotlib as mpl

mpl.rcParams["font.size"] = 16

mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["figure.autolayout"] = True

mpl.rcParams["axes.formatter.use_mathtext"] = True
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.labelpad"] = 8
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["axes.titleweight"] = "bold"
