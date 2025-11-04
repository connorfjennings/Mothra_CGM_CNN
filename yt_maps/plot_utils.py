import numpy as np
import matplotlib.pyplot as plt

def quiver_from_components(vx, vy, ax, step=1, scale=None,
                           clim=None, pivot='mid', key=None):
    """
    Quiver plot from x- and y-velocity components.

    Parameters
    ----------
    vx, vy : 2D arrays
        Velocity components on a regular grid. Shape must match.
        `vx` is the velocity in the x-direction, `vy` in the y-direction.
    step : int, optional
        Keep 1 in every `step` arrows along each axis (default 1 = keep all).
    scale : float or None, optional
        Passed to plt.quiver (arrow length scaling). None lets Matplotlib choose.
    cmap : str, optional
        Colormap for coloring by speed magnitude.
    clim : tuple or None, optional
        Color limits (vmin, vmax) for the magnitude.
    pivot : {'tip','mid','tail'}, optional
        Where the arrow pivots.
    key : tuple or None, optional
        If given, add a quiver key: (U, label) where U is the vector length in
        data units to illustrate, and label is a string like '100 km/s'.
    ax : matplotlib.axes.Axes or None
        Axes to draw on; if None, creates a new figure and axes.

    Returns
    -------
    fig, ax, q : Matplotlib figure, axes, and Quiver object
    """
    vx = np.asanyarray(vx)
    vy = np.asanyarray(vy)
    if vx.shape != vy.shape:
        raise ValueError(f"vx and vy must have same shape, got {vx.shape} vs {vy.shape}")

    # grid coordinates (assume unit spacing; replace with your x/y arrays if you have them)
    ny, nx = vx.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    # mask invalid points (NaNs/inf in either component)
    valid = np.isfinite(vx) & np.isfinite(vy)
    vx = np.where(valid, vx, np.nan)
    vy = np.where(valid, vy, np.nan)

    # subsample
    sl = (slice(None, None, step), slice(None, None, step))
    Xs, Ys = X[sl], Y[sl]
    Vx, Vy = vx[sl], vy[sl]

    # magnitude for coloring
    mag = np.hypot(Vx, Vy)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    q = ax.quiver(
        Xs, Ys, Vx, Vy,
        color='white',
        scale=scale,
        pivot=pivot,
        angles='xy',
        scale_units='xy',          # keeps arrow scale in data units
        width=0.003)                # a nice default; tweak as desired
    return q
