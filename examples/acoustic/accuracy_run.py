import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from examples.acoustic.Acoustic_codegen import Acoustic_cg
from examples.containers import IGrid, IShot
from opescibench import AxisScale, LinePlotter


# Velocity models
def smooth10(vel, shape):
    out = np.ones(shape)
    out[:, :] = vel[:, :]
    nx = shape[1]
    for a in range(5, nx-6):
        out[:, a] = np.sum(vel[:, a - 5:a + 5], axis=1) / 10
    return out

order = [2, 4, 6, 8]
size = 1600
scale = [1, 2, 4, 8, 16, 32, 64, 128]
grid = 1


# Set up the source as Ricker wavelet for f0
def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))

    return (1-2.*r**2)*np.exp(-r**2)

# Define geometry
dimensions = tuple([size] * 2)
origin = tuple([0.0] * len(dimensions))
spacing = tuple([grid] * len(dimensions))

vp = 1.5*np.ones(dimensions)

model = IGrid(origin, spacing, vp)
# Define seismic data.
data = IShot()
src = IShot()
f0 = .025
dt0 = model.get_critical_dt()
t0 = 0.0
tn = 400.0
nt = int(1+(tn-t0)/dt0)
final0 = (nt + 2) % 3
# Source geometry
time_series = np.zeros((nt, 1))

time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)

location = np.zeros((1, 2))
location[0, 0] = 800
location[0, 1] = 800
# location[0, 3] = 800


src.set_receiver_pos(location)
src.set_shape(nt, 1)
src.set_traces(time_series)
# src.set_time_axis(dt, tn)
# Receiver geometry
receiver_coords = np.zeros((201, 2))
receiver_coords[:, 0] = np.linspace(0, 1600, num=201)
receiver_coords[:, 1] = 200
# receiver_coords[:, 2] = 200
data.set_receiver_pos(receiver_coords)
data.set_shape(nt, 201)

Wave = Acoustic_cg(model, data, src, t_order=2, s_order=40, nbpml=40)
rec0, u0, gflopss, oi, timings = Wave.Forward()
error = np.zeros((3, 4))
time = np.zeros((3, 4))
scaleanno = np.zeros((3, 4))
for i in range(0, 3):
    for j in range(0, len(order)):
        # Define geometry
        scalei = scale[i + int(j/2)]  # / (2**(len(order) - j - 1))
        scaleanno[i, j] = scalei
        dimensions = tuple([size / scalei] * 2)
        origin = tuple([0.0] * len(dimensions))
        spacing = tuple([grid * scalei] * len(dimensions))
        vp = 1.5 * np.ones(dimensions)

        model = IGrid(origin, spacing, vp)

        f0 = .025
        dt = model.get_critical_dt()
        t0 = 0.0
        tn = 400.0
        nt = int(1 + (tn - t0) / dt)
        final = (nt + 2) % 3
        # Source geometry
        time_series = np.zeros((nt, 1))

        time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)

        Wave = Acoustic_cg(model, data, src, t_order=2, s_order=order[j], nbpml=40)
        rec, u, gflopss, oi, timings = Wave.Forward()
        error[i, j] = np.linalg.norm(u.data[final, 40:-40, 40:-40].reshape(-1)/np.linalg.norm(u.data[final, 40:-40, 40:-40].reshape(-1)) -
                                     u0.data[final0, 40:-40:scalei, 40:-40:scalei].reshape(-1)/np.linalg.norm(u0.data[final0, 40:-40:scalei, 40:-40:scalei].reshape(-1)))
        time[i, j] = timings['loop_body']
        # fig2 = plt.figure()
        # l = plt.imshow(np.transpose(u.data[final, 40:-40, 40:-40]/np.linalg.norm(u.data[final, 40:-40, 40:-40].reshape(-1))), vmin=-.1, vmax=.1, cmap=cm.gray, aspect=1)
        # fig2 = plt.figure()
        # l = plt.imshow(np.transpose(u0.data[final0, 40:-40:scalei, 40:-40:scalei]/np.linalg.norm(u0.data[final0, 40:-40:scalei, 40:-40:scalei].reshape(-1))), vmin=-.1, vmax=.1, cmap=cm.gray, aspect=1)
        # plt.show()


print(error)
print(time)

stylel = ('-^k', '-^b', '-^r', '-^g')

with LinePlotter(figname='MyPrettyPicture.pdf', plotdir='./',  xlabel='error') as plot:
    for i in range(0, 4):
        plot.add_line(error[:, i], time[:, i], label=('order %s' % order[i]),
                      annotations=[('dx = %s m' % sc) for sc in scaleanno[:, i]], style=stylel[i])

# fig2 = plt.figure()
# plt.loglog(error[:, 0], time[:, 0], error[:, 1], time[:, 1], error[:, 2], time[:, 2], error[:, 3], time[:, 3])
# plt.legend(['2nd order', '4th order', '6th order', '8th order'], loc='upper right')
# plt.show()