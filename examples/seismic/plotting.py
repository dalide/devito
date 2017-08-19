import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_perturbation(model, model1, colorbar=True):
    """
    Plot a two-dimensional velocity difference from two seismic :class:`Model`
    objects.

    :param model: :class:`Model` object of first velocity model.
    :param model1: :class:`Model` object of the second velocity model.
    :param source: Coordinates of the source point.
    :param receiver: Coordinates of the receiver points.
    """
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]
    dv = np.transpose(model.vp) - np.transpose(model1.vp)

    plt.figure()
    ax = plt.gca()
    plot = plt.imshow(dv, animated=True, cmap=cm.jet,
                      vmin=min(dv.reshape(-1)), vmax=max(dv.reshape(-1)),
                      extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')

    # Create aligned colorbar on the right
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity perturbation (km/s)')

    plt.show()


def plot_velocity(model, source=None, receiver=None, colorbar=True):
    """
    Plot a two-dimensional velocity field from a seismic :class:`Model`
    object. Optionally also includes point markers for sources and receivers.

    :param model: :class:`Model` object that holds the velocity model.
    :param source: Coordinates of the source point.
    :param receiver: Coordinates of the receiver points.
    """
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    plt.figure()
    ax = plt.gca()
    plot = plt.imshow(np.transpose(model.vp), animated=True, cmap=cm.jet,
                      vmin=min(model.vp.reshape(-1)),
                      vmax=max(model.vp.reshape(-1)),
                      extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')

    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(1e-3*receiver[:, 0], 1e-3*receiver[:, 1],
                    s=25, c='green', marker='D')

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(1e-3*source[:, 0], 1e-3*source[:, 1],
                    s=25, c='red', marker='o')

    # Create aligned colorbar on the right
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity (km/s)')

    plt.show()


def plot_shotrecord(rec, model, t0, tn, colorbar=True):
    """
    Plot a shot record (receiver values over time).

    :param rec: Receiver data with shape (time, points)
    :param model: :class:`Model` object that holds the velocity model.
    :param t0: Start of time dimension to plot
    :param tn: End of time dimension to plot
    """
    aspect = model.domain_size[0] / tn
    scale = np.max(rec) / 10.
    extent = [model.origin[0], model.origin[0] + 1e-3*model.domain_size[0],
              1e-3*tn, t0]

    plt.figure()
    ax = plt.gca()
    plot = plt.imshow(rec, vmin=-scale, vmax=scale, cmap=cm.gray,
                      aspect=aspect, extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Time (s)')

    # Create aligned colorbar on the right
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity (km/s)')

    plt.show()
