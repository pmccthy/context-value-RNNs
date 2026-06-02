"""
Visualisation utilities.
Author: patrick.mccarthy@dpag.ox.ac.uk
"""
from pathlib import Path

STYLE = Path(__file__).parent / "cxval.mplstyle"


def add_colorbar(ax, im, label="", size="5%", pad=0.05):
    """Attach a colorbar that is exactly the same height as ax.

    Uses make_axes_locatable to carve space from ax itself rather than
    shrinking a figure-level colorbar.  This prevents the bar from being
    shorter than the axes (which causes title overlap) regardless of the
    subplot's aspect ratio or layout engine.

    Args:
        ax:    The axes the colorbar belongs to.
        im:    The mappable returned by imshow / contourf / etc.
        label: Colorbar axis label.
        size:  Width of the colorbar as a fraction of ax (default "5%").
        pad:   Gap between ax and the colorbar axes (default 0.05 inches).

    Returns:
        The matplotlib Colorbar instance.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size=size, pad=pad)
    return ax.figure.colorbar(im, cax=cax, label=label)
