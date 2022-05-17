"""
Default values for plots and figures.
"""
cm = 1/2.54


def set_size(page_width, fraction=1, ratio=1., expected_with_cm=None):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    page_width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Width of figure
    if expected_with_cm is None:
        fig_width_pt = page_width * fraction
    else:
        assert False

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio

    return fig_width_in, fig_height_in


aspect = {
    're_fig3': 0.9,
    're_fig3_tall': 1.,
    're_fig3s2_boxplot': 1.2,
}

figsize = {
    're_fig3': (set_size(504, fraction=0.47)[0], 4.),
    're_fig3_tall': (set_size(504, fraction=0.47)[0], 5.),
    'psth_single': (set_size(504, fraction=0.35)[0], 1.3),
    'psth_single_large': (set_size(504, fraction=0.5)[0], 1.8),
    're_fig3s2_boxplot': set_size(504, fraction=0.55, ratio=0.9),
    're_fig5s1': set_size(504, fraction=0.9, ratio=0.35),
    're_fig3s2_weights': set_size(504, fraction=0.5, ratio=0.95),
    # 're_fig3s2_weights_landscape': set_size(504, fraction=0.9, ratio=0.4),
    're_fig3s2_weights_landscape': set_size(504, fraction=0.5, ratio=0.75),
    # 're_fig3s4_raster': set_size(504, fraction=0.5, ratio=1.4),
    're_fig3s4_raster': (set_size(504, fraction=0.5)[0], 4.),
    're_fig4s_traces': (set_size(504, fraction=1.)[0], 3.),
    're_fig5s1_scan': (set_size(504, fraction=0.48)[0], 3.),
    're_fig5s1_scan_med+std': (set_size(504, fraction=0.27)[0], 3.),
    'median_vs_weights': (set_size(504, fraction=0.6)[0], 2.),
}

colors = {
    'purple': '#867CE8',
    'turquoise': '#40E0D0',
    'orangegold': '#F9B233',
    'mediumturquoise': '#48D1CC',
}

ticksize = {
    'maps': 24,
    'maps2': 26
}
