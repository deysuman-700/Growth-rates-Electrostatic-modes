#!/usr/bin/env python3

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from numpy.polynomial.legendre import leggauss

from kappa_beam_plasma_plasma import (
    BeamParameters,
    Distribution,
    distribution_and_derivatives,
)


@lru_cache(maxsize=32)
def nodes(n):
    return leggauss(n)


def interval(n, a, b):
    x, w = nodes(n)
    return a + (b-a)*(x+1)/2, (b-a)*w/2


def rate_oblique(kpar, kperp, dist, rmax, nr=120, nz=160):
    if kperp <= 0 or kpar <= 1:
        raise ValueError('Require kperp > 0 and kpar > 1')

    r, wr = interval(nr, 0, rmax)
    theta, wt = interval(nz, 0, np.pi)

    R = r[:, None]
    b = kperp * R
    delta = (kpar - 1) * (kpar + 1)
    s = np.sqrt(b*b + delta*(1+R*R))

    zmin = (1+R*R-b*b)/(s+kpar*b)
    zmax = (s+kpar*b)/delta

    zmid = 0.5*(zmin+zmax)
    half = 0.5*(zmax-zmin)

    T = theta[None, :]
    Z = zmid + half*np.cos(T)

    gamma = np.hypot(Z, np.sqrt(1+R*R))
    h = (1+R*R)/(gamma+Z) - (kpar-1)*Z

    cosine = np.clip(h/b, -1.0, 1.0)

    _, fr, fz = distribution_and_derivatives(R, Z, dist)

    sinphi = np.sqrt(np.maximum(1-cosine*cosine, 1e-300))
    jac = half*np.sin(T)/sinphi

    integrand = (
        (2*gamma/kperp)
        * (kpar*fz+kperp*cosine*fr)
        * jac
    )

    J = np.sum(wr[:, None]*wt[None, :]*integrand)

    return float(
        0.5*np.pi*dist.params.alpha*J
        / (kpar*kpar+kperp*kperp)
    )


def rate_parallel(kpar, dist, rmax, nr=240):
    if kpar <= 1:
        return 0.0

    r, wr = interval(nr, 0.0, rmax)

    delta = np.expm1(2*np.log(kpar))
    z = np.sqrt((1+r*r)/delta)
    gamma = kpar*z

    _, _, fz = distribution_and_derivatives(r, z, dist)

    J = 2*np.pi*np.sum(
        wr*r*gamma**3/(kpar*(1+r*r))*kpar*fz
    )

    return float(
        0.5*np.pi*dist.params.alpha*J/kpar**2
    )


def plot_growth_map(
    outdir,
    kpar,
    kperp,
    growth,
    optimum_kpar,
    optimum_kperp,
):
    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.ticker import FormatStrFormatter

    try:
        import scienceplots
        plt.style.use('science')
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(6, 4))

    vmax = float(np.max(np.abs(growth)))

    norm = SymLogNorm(
        linthresh=0.01*vmax,
        linscale=1.0,
        vmin=-vmax,
        vmax=vmax,
        base=10,
    )

    xvalues = (kpar-1)*1e9

    mesh = ax.pcolormesh(
        xvalues,
        kperp,
        growth,
        shading='nearest',
        cmap='RdBu_r',
        norm=norm,
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        pad=0.025,
        fraction=0.045,
        shrink=0.92,
    )

    cbar.set_label(
        r'$\zeta/\omega_{pe}$',
        fontsize=12,
    )

    cbar.set_ticks([
        -vmax,
        -0.1*vmax,
        0.0,
        0.1*vmax,
        vmax,
    ])

    cbar.ax.yaxis.set_major_formatter(
        FormatStrFormatter('%.1e')
    )

    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=9)

    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=1e-5)

    ax.set_xlabel(
        r'$(k_\parallel c/\omega_{pe}-1)\times10^9$',
        fontsize=13,
    )

    ax.set_ylabel(
        r'$k_\perp c/\omega_{pe}$',
        fontsize=13,
    )

    xstar = (optimum_kpar-1)*1e9
    ystar = optimum_kperp

    ax.plot(
        xstar,
        ystar,
        marker='*',
        markersize=11,
        color='black',
        markeredgecolor='white',
        markeredgewidth=0.8,
        linestyle='none',
        zorder=10,
    )

    xmid = np.sqrt(xvalues.min()*xvalues.max())

    if xstar >= xmid:
        offset = (-10, 10)
        horizontal_alignment = 'right'
    else:
        offset = (10, 10)
        horizontal_alignment = 'left'

    ax.annotate(
        'Maximum growth',
        xy=(xstar, ystar),
        xytext=offset,
        textcoords='offset points',
        fontsize=10,
        ha=horizontal_alignment,
        va='bottom',
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.85,
            pad=2,
        ),
        zorder=11,
    )

    ax.tick_params(
        axis='both',
        which='major',
        labelsize=10,
    )

    ax.grid(False)

    fig.tight_layout()

    fig.savefig(
        outdir/'growth_map_astrophysics.png',
        dpi=800,
        bbox_inches='tight',
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--quick',
        action='store_true',
    )

    parser.add_argument(
        '--outdir',
        type=Path,
        default=Path('astrophysical_kappa_results'),
    )

    args = parser.parse_args()

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    p = BeamParameters(
        alpha=5e-17,
        p0=1e5,
        sigma_parallel=5e4,
        sigma_perp_1d=10,
    )

    dist = Distribution('kappa', 2.0, p)

    tests = [
        (1+10e-9, 1e-4),
        (1+50e-9, 1e-3),
        (1+200e-9, 1e-2),
    ]

    rows = []

    orders = (
        [(80, 120), (160, 240)]
        if args.quick
        else [(80, 120), (160, 240), (320, 480)]
    )

    for kp, kt in tests:
        for nr, nz in orders:
            value = rate_oblique(
                kp,
                kt,
                dist,
                200.0,
                nr=nr,
                nz=nz,
            )

            print(
                f'Kpar={kp:.12f} '
                f'Kperp={kt:g} '
                f'nr={nr} '
                f'nz={nz} '
                f'growth={value:.10e}',
                flush=True,
            )

            rows.append({
                'Kpar': kp,
                'Kperp': kt,
                'nr': nr,
                'nz': nz,
                'growth': value,
            })

    with (args.outdir/'convergence.csv').open(
        'w',
        newline='',
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0],
        )
        writer.writeheader()
        writer.writerows(rows)

    kpar = 1 + np.geomspace(
        1e-10,
        1e-4,
        14 if args.quick else 38,
    )

    kperp = np.concatenate((
        [0.0],
        np.geomspace(
            1e-5,
            0.6,
            14 if args.quick else 38,
        ),
    ))

    growth = np.empty(
        (len(kperp), len(kpar)),
        dtype=float,
    )

    for i, kt in enumerate(kperp):
        for j, kp in enumerate(kpar):
            if kt == 0:
                growth[i, j] = rate_parallel(
                    kp,
                    dist,
                    200.0,
                    nr=160 if args.quick else 300,
                )
            else:
                growth[i, j] = rate_oblique(
                    kp,
                    kt,
                    dist,
                    200.0,
                    nr=110 if args.quick else 180,
                    nz=170 if args.quick else 280,
                )

        print(
            f'map row {i+1}/{len(kperp)}, '
            f'max={np.max(growth[i]):.4e}',
            flush=True,
        )

    np.savez(
        args.outdir/'map.npz',
        kpar=kpar,
        kperp=kperp,
        growth=growth,
    )

    index = np.unravel_index(
        np.argmax(growth),
        growth.shape,
    )

    print(
        'Grid maximum:',
        'Kpar', kpar[index[1]],
        'Kperp', kperp[index[0]],
        'growth', growth[index],
    )

    def objective(x):
        kpar_value = 1 + 10**x[0]
        kperp_value = 10**x[1]

        return -rate_oblique(
            kpar_value,
            kperp_value,
            dist,
            200.0,
            nr=180,
            nz=280,
        )*1e14

    if index[0]:
        start = [
            np.log10(kpar[index[1]]-1),
            np.log10(kperp[index[0]]),
        ]
    else:
        start = [-6, -2]

    optimum = minimize(
        objective,
        start,
        method='Nelder-Mead',
        options={
            'maxiter': 120,
            'xatol': 1e-5,
            'fatol': 1e-7,
        },
    )

    optimum_kpar = 1 + 10**optimum.x[0]
    optimum_kperp = 10**optimum.x[1]

    optimum_growth = rate_oblique(
        optimum_kpar,
        optimum_kperp,
        dist,
        200.0,
        nr=320,
        nz=480,
    )

    print(
        'Local optimized maximum:',
        optimum_kpar,
        optimum_kperp,
        optimum_growth,
    )

    print(
        'Propagation angle (deg):',
        np.degrees(
            np.arctan2(
                optimum_kperp,
                optimum_kpar,
            )
        ),
    )

    plot_growth_map(
        args.outdir,
        kpar,
        kperp,
        growth,
        optimum_kpar,
        optimum_kperp,
    )

    print(
        'Figure saved:',
        args.outdir/'growth_map_astrophysics.png',
    )


if __name__ == '__main__':
    main()
