import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc, cm
from matplotlib.colors import ListedColormap
import numpy as np

# default parameters for all plots
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = ''.join([r'\usepackage{{amsmath}}',
                                               r'\usepackage{mathtools}',
                                               r'\usepackage{helvet}'])

plt.rcParams.update({
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "grid.color": (1, 1, 1, 0),
    "font.family": "sans-serif",  # use serif/main font for text elements
    "font.sans-serif": ['Helvetica'],
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
    "pgf.preamble": "\n".join([
         r'\usepackage{{amsmath}}',            # load additional packages
         r'\usepackage{mathtools}',   # unicode math setup
         r'\usepackage{helvet}'
    ])
})

# where to save figures
fig_path = "../figures/final/"

# define consistent colors and color maps for all figures
inhibitory_blue = '#529fb9'
excitatory_red = '#ec523d'
panel_gray = '#525252'
blues = cm.get_cmap('Blues', 10)
blues = blues(np.linspace(0.5, 0.5, 10))
blue_cmap = ListedColormap(blues)
reds = cm.get_cmap('Reds', 10)
reds = reds(np.linspace(0.5, 0.5, 10))
red_cmap = ListedColormap(reds)
inhibitory_cmap = plt.get_cmap('gist_earth')
excitatory_cmap = plt.get_cmap('autumn')

# font sizes
fs1 = 5.1
fs2 = 6.1
fs3 = 7.1
fs4 = 9.1

panel_prms = {'fontweight': 'bold', 'color': panel_gray, 'fontsize': fs4}
