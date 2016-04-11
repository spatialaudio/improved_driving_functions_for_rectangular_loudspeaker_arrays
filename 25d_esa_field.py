""" Generates Figure 4a of the paper
    Sascha Spors, Frank Schultz, and Till Rettberg. Improved Driving Functions
    for Rectangular Loudspeaker Arrays Driven by Sound Field Synthesis. In
    German Annual Conference on Acoustics (DAGA), March 2016.

    2.5D synthesis of a point source with the equivalent scattering approach
    using a edge-shaped secondary source distribution

    (c) Sascha Spors 2016, MIT Licence
"""
import numpy as np
import matplotlib.pyplot as plt
import sfs


# simulation switches
save_figures = False
wfs = False  # WFS or ESA

# simulation parameters
dx = 0.003  # secondary source distance
N = 10000  # number of secondary sources for one array
f = 500  # frequency
Nc = 350  # max circular harmonics
omega = 2 * np.pi * f  # angular frequency
src_angles = [180-45]
R = 4
xref = [2, -2, 0]  # reference point

grid = sfs.util.xyz_grid([0, 4], [-4, 0], 0, spacing=0.02)


def compute_sound_field(x0, n0, a0, omega, angle):
    xs = xref + R * np.asarray(sfs.util.direction_vector(np.radians(angle), np.radians(90)))

    if wfs:
        d = sfs.mono.drivingfunction.wfs_25d_point(omega, x0, n0, xs, xref=xref)
        a = sfs.mono.drivingfunction.source_selection_point(n0, x0, xs)
    else:
        d = sfs.mono.drivingfunction.esa_edge_25d_point(omega, x0, xs, xref=xref, Nc=Nc)    
        a = np.ones(d.shape[0])

    twin = sfs.tapering.none(a)

    p = sfs.mono.synthesized.generic(omega, x0, n0, d * twin * a0, grid,
                                     source=sfs.mono.source.point)

    return p, twin, xs


def plot_objects(ax, xs):
    ax.plot((0, 0), (-4.2, 0), 'k-', lw=2)
    ax.plot((0, 4.2), (0, 0), 'k-', lw=2)
    plt.annotate('4m', (0, 0.27), (4.1, 0.2), arrowprops={'arrowstyle': '<->'})

    sfs.plot.virtualsource_2d(xs, type='point', ax=ax)
    sfs.plot.reference_2d(xref, ax=ax)


def plot_sound_field(p, xs, twin):

    plt.style.use(('paper.mplstyle', 'paper_box.mplstyle'))
    fig = plt.figure()
    ax1 = fig.add_axes([0.0, 0.0, 0.7, 1])
    sfs.plot.soundfield(p, grid, xnorm=None, colorbar=False, vmax=1.5, vmin=-1.5, ax=ax1)
    plot_objects(ax1, xs)
    plt.axis([-1.1, 4.2, -4.2, 1.1])
    plt.axis('off')

    myfig = plt.gcf()
    plt.show()
    if save_figures:
        if wfs:
            myfig.savefig('../paper/figs/edge_wfs_25d_point_%dHz_%d.pdf' % (f, src_angles[n]), dpi=300)
        else:
            myfig.savefig('../paper/figs/edge_esa_25d_point_%dHz_%d.pdf' % (f, src_angles[n]), dpi=300)


def plot_sound_field_level(p, xs, twin):

    plt.style.use(('paper.mplstyle', 'paper_box.mplstyle'))
    fig = plt.figure()
    ax1 = fig.add_axes([0.0, 0.0, 0.7, 1])
    im = sfs.plot.level(p, grid, xnorm=None, colorbar=False, cmap=plt.cm.viridis, vmax=3, vmin=-3, ax=ax1)
    CS = plt.contour(sfs.util.db(p), 1, levels=[0], origin='lower', linewidths=2, extent=(0, 4, -4, 0), colors='w', alpha=.5)
    plt.clabel(CS, [0], inline=1, fmt='%1.1f dB', fontsize=8, rightside_up=1)
    zc = CS.collections[0]
    plt.setp(zc, linewidth=0.5)

    plot_objects(plt.gca(), xs)
    plt.annotate('4m', (-2.5, 2), (-2.75, -2.4), arrowprops={'arrowstyle': '<->'}) 

    plt.axis([-1.1, 4.2, -4.2, 1.1])
    plt.axis('off')

    ax2 = fig.add_axes([0.55, -0.05, 0.25, .95])
    plt.axis('off')
    cbar = plt.colorbar(im, ax=ax2, shrink=.6)
    cbar.set_label('relative level (dB)', rotation=270, labelpad=10)
    cbar.set_ticks(np.arange(-3, 4))

    myfig = plt.gcf()
    plt.show()
    if save_figures:
        if wfs:
            myfig.savefig('../paper/figs/edge_wfs_25d_point_%dHz_%d_L.pdf' % (f, src_angles[n]), dpi=300)
        else:
            myfig.savefig('../paper/figs/edge_esa_25d_point_%dHz_%d_L.pdf' % (f, src_angles[n]), dpi=300)


# get secondary source positions
x0, n0, a0 = sfs.array.rounded_edge(N, 0, dx, orientation=[0, -1, 0])

# compute field at the given positions for given virutal source
p = []
trajectory = []
lsactive = []

for angle in src_angles:
    tmp, twin, xs = compute_sound_field(x0, n0, a0, omega, angle)
    p.append(tmp)
    trajectory.append(xs)
    lsactive.append(twin)

p = np.asarray(p)
trajectory = np.asarray(trajectory)
lsactive = np.asarray(lsactive)


# plot synthesized sound field for multiple virtual source position
normalization = np.abs(sfs.mono.source.point(omega, xs, [0, 0, 0], xref))

for n in range(0, p.shape[0]):
    plot_sound_field(p[n, :, :]/normalization, trajectory[n, :], lsactive[n, :])
    plot_sound_field_level(p[n, :, :]/normalization, trajectory[n, :], lsactive[n, :])
