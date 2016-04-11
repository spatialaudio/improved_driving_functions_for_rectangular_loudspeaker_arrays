""" Generates Figure 1b of the paper
    Sascha Spors, Frank Schultz, and Till Rettberg. Improved Driving Functions
    for Rectangular Loudspeaker Arrays Driven by Sound Field Synthesis. In
    German Annual Conference on Acoustics (DAGA), March 2016.

    2D scattering of a line source at an semi-infinte edge.

    (c) Sascha Spors 2016, MIT Licence
"""
import numpy as np
import sfs
import matplotlib.pyplot as plt

f = 500  # frequency
omega = 2 * np.pi * f  # angular frequency
alpha = 270/180*np.pi  # outer angle of edge
xs = [-2, 2, 0]  # position of line source
Nc = 400  # max number of circular harmonics


# compute field
grid = sfs.util.xyz_grid([-3, 5], [-5, 3], 0, spacing=0.02)
p = sfs.mono.source.line_dirichlet_edge(omega, xs, grid, Nc=Nc)


# plot field
plt.style.use(('paper.mplstyle', 'paper_box.mplstyle'))
fig = plt.figure()
ax = fig.gca()
sfs.plot.soundfield(30*p, grid, colorbar=False)
ax.plot((0, 0), (-5, 0), 'k-', lw=2)
ax.plot((0, 5), (0, 0), 'k-', lw=2)
plt.axis('off')
fig.savefig('../paper/figs/scattering_edge_500Hz.pdf')
