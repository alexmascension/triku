import scanpy as sc
import pandas as pd
import numpy as np

from bokeh.plotting import figure
import bokeh.io as io
from bokeh.models import ColumnDataSource, LinearColorMapper

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ..logg import logger
from ..utils import return_proportion_zeros, return_mean, check_count_mat
from ..utils._general_utils import get_arr_counts_genes, get_dict_triku



def return_carto_cmap(cmap_str):
    dict_cmaps = {'Burg': ['#ffc6c4', '#f4a3a8', '#e38191', '#cc607d', '#ad466c', '#8b3058', '#672044'],
                  'BurYl': ['#fbe6c5', '#f5ba98', '#ee8a82', '#dc7176', '#c8586c', '#9c3f5d', '#70284a'],
                  'RedOr': ['#f6d2a9', '#f5b78e', '#f19c7c', '#ea8171', '#dd686c', '#ca5268', '#b13f64'],
                  'OrYel': ['#ecda9a', '#efc47e', '#f3ad6a', '#f7945d', '#f97b57', '#f66356', '#ee4d5a'],
                  'Peach': ['#fde0c5', '#facba6', '#f8b58b', '#f59e72', '#f2855d', '#ef6a4c', '#eb4a40'],
                  'PinkYl': ['#fef6b5', '#ffdd9a', '#ffc285', '#ffa679', '#fa8a76', '#f16d7a', '#e15383'],
                  'Mint': ['#e4f1e1', '#b4d9cc', '#89c0b6', '#63a6a0', '#448c8a', '#287274', '#0d585f'],
                  'BluGrn': ['#c4e6c3', '#96d2a4', '#6dbc90', '#4da284', '#36877a', '#266b6e', '#1d4f60'],
                  'DarkMint': ['#d2fbd4', '#a5dbc2', '#7bbcb0', '#559c9e', '#3a7c89', '#235d72', '#123f5a'],
                  'Emrld': ['#d3f2a3', '#97e196', '#6cc08b', '#4c9b82', '#217a79', '#105965', '#074050'],
                  'BluYl': ['#f7feae', '#b7e6a5', '#7ccba2', '#46aea0', '#089099', '#00718b', '#045275'],
                  'Teal': ['#d1eeea', '#a8dbd9', '#85c4c9', '#68abb8', '#4f90a6', '#3b738f', '#2a5674'],
                  'TealGrn': ['#b0f2bc', '#89e8ac', '#67dba5', '#4cc8a3', '#38b2a3', '#2c98a0', '#257d98'],
                  'Purp': ['#f3e0f7', '#e4c7f1', '#d1afe8', '#b998dd', '#9f82ce', '#826dba', '#63589f'],
                  'PurpOr': ['#f9ddda', '#f2b9c4', '#e597b9', '#ce78b3', '#ad5fad', '#834ba0', '#573b88'],
                  'Sunset': ['#f3e79b', '#fac484', '#f8a07e', '#eb7f86', '#ce6693', '#a059a0', '#5c53a5'],
                  'Magenta': ['#f3cbd3', '#eaa9bd', '#dd88ac', '#ca699d', '#b14d8e', '#91357d', '#6c2167'],
                  'SunsetDark': ['#fcde9c', '#faa476', '#f0746e', '#e34f6f', '#dc3977', '#b9257a', '#7c1d6f'],
                  'BrwnYl': ['#ede5cf', '#e0c2a2', '#d39c83', '#c1766f', '#a65461', '#813753', '#541f3f'],
                  'ArmyRose': ['#798234', '#a3ad62', '#d0d3a2', '#fdfbe4', '#f0c6c3', '#df91a3', '#d46780'],
                  'Fall': ['#3d5941', '#778868', '#b5b991', '#f6edbd', '#edbb8a', '#de8a5a', '#ca562c'],
                  'Geyser': ['#008080', '#70a494', '#b4c8a8', '#f6edbd', '#edbb8a', '#de8a5a', '#ca562c'],
                  'Temps': ['#009392', '#39b185', '#9ccb86', '#e9e29c', '#eeb479', '#e88471', '#cf597e'],
                  'TealRose': ['#009392', '#72aaa1', '#b1c7b3', '#f1eac8', '#e5b9ad', '#d98994', '#d0587e'],
                  'Tropic': ['#009B9E', '#42B7B9', '#A7D3D4', '#F1F1F1', '#E4C1D9', '#D691C1', '#C75DAB'],
                  'Earth': ['#A16928', '#bd925a', '#d6bd8d', '#edeac2', '#b5c8b8', '#79a7ac', '#2887a1'],
                  }

    if cmap_str in dict_cmaps.keys():
        return dict_cmaps[cmap_str]
    elif cmap_str[:3] == 'inv' and cmap_str[3:] in dict_cmaps.keys():
        return dict_cmaps[cmap_str[3:]][::-1]
    else:
        logger.warning(
            '{} colormap name not found. Ignore this if you are using a matplotlib colormap.'.format(cmap_str))
        return []


def entropy(object_triku: [sc.AnnData, pd.DataFrame, str], dict_triku: dict = None, dict_triku_path: str = '',
            backend: str = 'bokeh', size_small: float = 3, size_large: float = 6, alpha_small: float = 3, alpha_large: float = 6,
            cmap_entropy: [list, str] = 'invSunsetDark', return_fig: bool = False, show: bool = True,
            save_path: str = '', line_color: str = '#000000', line_alpha: float = 0.1, ax: plt.axes = None,
            figsize: tuple = (10, 5), x_label: str = '', y_label: str = ''):
    """
    Plots the mean expression VS percentage of 0, adding information about the entropy and the
    genes selected by `tl.triku()`. If the object is an annData, information from the plot can be extracted directly
    from the annData. Else, it has to be added to `dict_triku`.
    The type of plot is a scatter plot where not selected genes appear smaller and more transparent, while
    selected genes appear larger and less transparent. Each dot's color is represented by the entropy of the gene.

    Parameters
    ----------
    object_triku : scanpy.AnnData or pandas.DataFrame or str
        Object with count matrix. If `pandas.DataFrame`, rows are cells and columns are genes.
        If str, path to the annData file or pandas DataFrame.
    dict_triku : dict
        `dict_triku` object from `tl.triku`.
    dict_triku_path : str
        Path to the dict_triku objects if called from the CLI. For instance, if /filesdir/experiment_2_entropy.txt and
        /filesdir/experiment_2_selected_genes.txt are generated, then dict_triku_path is /filesdir/experiment_2.
    backend : str
        Option to plot ['bokeh', 'matplotlib']. Uses bokeh, which outputs a html, or matplotlib, which outputs an
        image.
    size_small : float
        Dot size for not selected genes.
    size_large : float
        Dot size for selected genes.
    alpha_small : float
        Dot alpha for not selected genes.
    alpha_large : float
        Dot alpha for selected genes.
    cmap_entropy : [list, str]
        List of colors, or colormap name, to represent entropy values.
    return_fig : bool
        If `True`, returns the figure (bokeh or matplotlib).
    show : bool
        Shows the plot on screen.
    save_path : str
        Saves the figure to the path.
    line_color : str
        Color of dot line.
    line_alpha : float
        Alpha of dot line
    ax : matplolix.Axis
        Axis the plot will be saved to (matplotlib)
    figsize : tuple(int, int)
        Size of figure (matplotlib)
    x_label : str
        Label of x axis.
    y_label : str
        Label of y axis.

    Returns
    -------
    fig :
        Figure.
    """
    # Check type of object and return the matrix as corresponded
    arr_counts, arr_genes = get_arr_counts_genes(object_triku)

    # Initialize dict triku based on the object type
    dict_triku = get_dict_triku(dict_triku, dict_triku_path, object_triku)

    if backend not in ['bokeh', 'matplotlib']:
        logger.error('backend must be "bokeh" or "matplotlib".')
        TypeError('backend must be "bokeh" or "matplotlib".')

    # Calculate the mean and proportion of zeros
    check_count_mat(arr_counts)
    mean, prop_0 = return_mean(arr_counts), return_proportion_zeros(arr_counts)

    # The the cmap from CartoColors if it exists
    if isinstance(cmap_entropy, str):
        carto_cmap = return_carto_cmap(cmap_entropy)
        if len(carto_cmap) > 0: cmap_entropy = carto_cmap

    # Do the plotting
    if backend == 'bokeh':
        logger.info("Doing plot with backend 'bokeh")
        # Create the figure with the tools
        fig = figure(tools='reset,box_zoom', tooltips=[('Gene', "@genes"), ('% Zeros', "@zero_per"),
                                                        ('% Entropy', "@entropy")])

        fig.xaxis.axis_label = x_label
        fig.yaxis.axis_label = y_label

        # Create a ColorMapper object to do a continuous mapping between the colormap and the data
        color_mapper = LinearColorMapper(palette=cmap_entropy, low=min(dict_triku['triku_entropy'].values()),
                                         high=max(dict_triku['triku_entropy'].values()))

        triku_genes_idx = np.argwhere(arr_genes.isin(dict_triku['triku_selected_genes'])).flatten()

        arr_sizes, arr_alphas = np.full(len(mean), size_small), np.full(len(mean), alpha_small)

        arr_sizes[triku_genes_idx], arr_alphas[triku_genes_idx] = size_large, alpha_large

        cds = ColumnDataSource({'x': np.log10(mean), 'y': prop_0, 'genes': arr_genes,
                                'entropy': list(dict_triku['triku_entropy'].values()),
                                'alpha': arr_alphas, 'size': arr_sizes,
                                'zero_per': 100 * prop_0})

        fig.scatter('x', 'y', source=cds, fill_alpha='entropy', color={'field': 'entropy', 'transform': color_mapper},
                  size='size', line_alpha=line_alpha, line_color=line_color)

        if show:
            io.show(fig)
        if save_path != '':
            logger.info("Saving figure in {}.".format(save_path))
            io.save(fig, save_path)
        if return_fig:
            return fig

    elif backend == 'matplotlib':
        logger.info("Doing plot with backend 'matplotlib")
        fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot(111)

        cmax = LinearSegmentedColormap.from_list('trikucmap', cmap_entropy)

        triku_genes_idx = np.argwhere(arr_genes.isin(dict_triku['triku_selected_genes'])).flatten()
        inverse_triku_genes_idx = ~np.in1d(range(len(arr_genes)), triku_genes_idx)

        arr_entropy = np.array(list((dict_triku['triku_entropy'].values())))

        print(np.log10(mean[inverse_triku_genes_idx]), prop_0[inverse_triku_genes_idx])
        ax.scatter(np.log10(mean[inverse_triku_genes_idx]), prop_0[inverse_triku_genes_idx],
                   c=arr_entropy[inverse_triku_genes_idx], cmap=cmax, s=size_small, alpha=alpha_small,
                   edgecolors=line_color, linewidths=0.05 * size_small)
        ax.scatter(np.log10(mean[triku_genes_idx]), prop_0[triku_genes_idx],
                   c=arr_entropy[triku_genes_idx], cmap=cmax, s=size_large, alpha=alpha_large, edgecolors=line_color,
                   linewidths=0.05 * size_small)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        plt.tight_layout()

        if show:
            plt.show()
        if save_path != '':
            logger.info("Saving figure in {}.".format(save_path))
            fig.savefig(save_path)
        if return_fig:
            return fig

