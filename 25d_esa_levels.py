""" Generates Figure 4b of the paper
    Sascha Spors, Frank Schultz, and Till Rettberg. Improved Driving Functions
    for Rectangular Loudspeaker Arrays Driven by Sound Field Synthesis. In
    German Annual Conference on Acoustics (DAGA), March 2016.

    2.5D synthesis of a point source with the equivalent scattering approach
    using a edge-shaped secondary source distribution. The level at the
    reference point is evaluated for various virtual source positions.

    (c) Sascha Spors 2016, MIT Licence
"""
import numpy as np
import matplotlib.pyplot as plt
import sfs

# simulation switches
save_figures = False

# simulation parameters
N = 6000
dx = 0.005
Nc = 220  # max circular harmonics
f = np.array([500, ])  # frequencies
omega = 2 * np.pi * f  # angular frequency
angle = 135  # virtual source angle
rs = np.linspace(1, 10, num=10)  # virtual source distances
xref = [2, -2, 0]  # reference point

grid = sfs.util.xyz_grid(xref[0], xref[1], 0, spacing=1)  # evaluated grid


def compute_sound_field(x0, n0, a0, omega, R):
    xs = R * np.asarray(sfs.util.direction_vector(np.radians(angle), np.radians(90)))

    d = sfs.mono.drivingfunction.esa_edge_25d_point(omega, x0, xs, xref=xref, Nc=Nc)
    a = np.ones(d.shape[0])

    twin = sfs.tapering.none(a)

    p = sfs.mono.synthesized.generic(omega, x0, n0, d * twin * a0, grid,
                                     source=sfs.mono.source.point)
      
    return p, twin, xs


def iterate_compute_sound_field():
    # compute field at the given positions for given frequencies
    p = []

    for Ri in rs:
        for omegan in omega:
            tmp, twin, xs = compute_sound_field(x0, n0, a0, omegan, Ri)
            p.append(tmp)

    p = np.asarray(p)

    return p


# get secondary source positions
x0, n0, a0 = sfs.array.rounded_edge(N, 0, dx, orientation=[0, -1, 0])

# compute level at reference point
pESA = iterate_compute_sound_field()


# plot the resulting amplitudes
ps = np.zeros(len(rs))
for n in range(len(rs)):
    xs = rs[n] * np.asarray(sfs.util.direction_vector(np.radians(angle), np.radians(90)))
    ps[n] = np.abs(sfs.mono.source.point(omega, xs, [0, 0, 0], xref))

normalization = np.abs(ps[1])

plt.style.use(('paper.mplstyle', 'paper_box3.mplstyle'))
fig = plt.figure()
ax = plt.gca()
plt.plot(rs, sfs.util.db(pESA.T/normalization), 'b-', label='2.5D ESA')
plt.plot(rs, sfs.util.db(ps.T/normalization), 'g--', label='point source')
plt.xlabel(r'distance $r_s$ in m')
plt.ylabel('relative level in dB')
plt.axis([rs[0], rs[-1], -10, 3])
plt.grid()
plt.legend(loc='upper right')

plt.show()
if save_figures:
    fig.savefig('../paper/figs/esa_25d_line_ref_level.pdf', dpi=300)
