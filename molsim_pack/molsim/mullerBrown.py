import numpy as np
import numba
import matplotlib.pyplot as plt

# Define the coefficients
A = np.array([-200.0, -100.0, -170.0, 15.0])
a = np.array([-1.0, -1.0, -6.5, 0.7])
b = np.array([0.0, 0.0, 11.0, 0.6])
c = np.array([-10.0, -10.0, -6.5, 0.7])
x0 = np.array([1.0, 0.0, -0.5, -1.0])
y0 = np.array([0.0, 0.5, 1.5, 1.0])


@numba.vectorize(["float64(float64, float64)"])
def mullerBrownPotential(x, y):
    zi = A * np.exp(a * (x - x0) ** 2 + b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
    return np.sum(zi)


@numba.vectorize(["float64(float64, float64)"])
def mullerBrownGradient_dx(x, y):
    zi = A * np.exp(a * (x - x0) ** 2 + b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
    dx = np.sum((2 * a * (x - x0) + b * (y - y0)) * zi)
    return dx


@numba.vectorize(["float64(float64, float64)"])
def mullerBrownGradient_dy(x, y):
    zi = A * np.exp(a * (x - x0) ** 2 + b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
    dy = np.sum((b * (x - x0) + 2 * c * (y - y0)) * zi)
    return dy


def mullerBrownHeatmap(N, xrange=(-1.6, 1.4), yrange=(-0.7, 2.1)):
    x = np.linspace(*xrange, N)
    y = np.linspace(*yrange, N)
    X, Y = np.meshgrid(x, y)
    Z = mullerBrownPotential(X, Y)
    return X, Y, Z


@numba.njit
def mullerBrownPotentialAndGradient(pos):
    return (
        mullerBrownPotential(pos[0], pos[1]),
        mullerBrownGradient_dx(pos[0], pos[1]),
        mullerBrownGradient_dy(pos[0], pos[1]),
    )


def plot_muller_brown_heatmap(ax=None, vmin=-150, vmax=300):
    """
    Plots a contour map of the Müller-Brown potential with discrete levels, resembling a rice field.

    Parameters:
    - ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None, a new figure and axes are created.
    - vmin (float): Minimum value for potential. Default is -150.
    - vmax (float): Maximum value for potential. Default is 300.
    - levels (int or sequence): Number of discrete levels or specific level values. Default is 20.
    """

    X, Y, Z = mullerBrownHeatmap(500)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    levels = np.linspace(vmin, vmax, 30)
    im = ax.contourf(X, Y, Z, levels=levels, cmap="cividis", extend="both")
    ax.set_xlim(-1.6, 1.4)
    ax.set_ylim(-0.7, 2.1)

    # Add a colorbar to show the potential scale
    cbar = plt.colorbar(im)
    cbar.set_label(r"Potential $U_{MB}$(x, y)")

    # Define evenly spaced tick values
    tick_values = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{tick:.0f}" for tick in tick_values])  # Format ticks as integers

    # Label the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(False)
