#!/usr/bin/env python3

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
try:
    import scienceplots 
    plt.style.use('science')
except ImportError:
   
    plt.style.use('default')
from kappa_beam_plasma_plasma import (
    BeamParameters, Distribution, parallel_marginal,
    transverse_component_marginal, scan_growth_rate, gamma_over_omega_pe,
)
def make_dists(params, kappas):
    ds = [(f'kappa={k:g}', Distribution('kappa', float(k), params)) for k in kappas]
    ds.append(('Gaussian', Distribution('gaussian', None, params)))
    ds.append(('Maxwell-Jüttner', Distribution('maxwell_juttner', None, params)))
    return ds
def scan_axes(params, nkpar, nkperp, kperp_max=3.0):
    centre = params.k_parallel_resonant_centre
    dk = params.dk_parallel_per_sigma
    kpar = np.linspace(centre - 4.0*dk, centre + 4.0*dk, nkpar)
    
    k_transition = min(float(kperp_max), max(0.1, 6.0 * params.sigma_perp_1d / params.p0))
    n_linear = max(5, int(round(0.55 * nkperp)))
    n_linear = min(n_linear, nkperp)
    if k_transition >= float(kperp_max) or n_linear == nkperp:
        kperp = np.linspace(0.0, float(kperp_max), nkperp)
    else:
        k1 = np.linspace(0.0, k_transition, n_linear)
        k2 = np.geomspace(k_transition, float(kperp_max), nkperp - n_linear + 1)[1:]
        kperp = np.concatenate([k1, k2])
    if np.min(kpar) <= 1.0:
        raise ValueError('This script requires all scanned K_parallel > 1. Increase p0 or narrow the scan.')
    return kpar, kperp
def save_csv(path, rows, fieldnames):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
def plot_marginals(outdir, params, dists):
    y = np.linspace(-8, 8, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    for label, dist in dists:
        legend_label = format_distribution_title(label)
        
        if label == 'Maxwell-Jüttner':
            linestyle, linewidth = '--', 1.9
        elif label == 'Gaussian':
            linestyle, linewidth = ':', 1.7
        else:
            linestyle, linewidth = '-', 1.3
        axes[0].semilogy(
            y, parallel_marginal(y, dist), label=legend_label,
            linestyle=linestyle, linewidth=linewidth
        )
        axes[1].semilogy(
            y, transverse_component_marginal(y, dist), label=legend_label,
            linestyle=linestyle, linewidth=linewidth
        )
    axes[0].set_xlabel(r'$(p_\parallel-p_0)/\sigma_\parallel$', fontsize=13)
    axes[0].set_ylabel(
        r'$\frac{\sigma_\parallel}{n_b}\int f\,d^2p_\perp$', fontsize=12
    )
    axes[1].set_xlabel(r'$p_x/\sigma_{\perp,1D}$', fontsize=13)
    axes[1].set_ylabel(
        r'$\frac{\sigma_{\perp,1D}}{n_b}\int f\,dp_y\,dp_\parallel$',
        fontsize=11
    )
    for ax in axes:
        ax.grid(False)
        ax.set_ylim(1e-7, 2)
        ax.tick_params(axis='both', labelsize=10)
    axes[0].legend(fontsize=8.5)
    fig.savefig(outdir/'01_distribution_function.png', dpi=800)
    plt.close(fig)

def format_distribution_title(label):
    """Format distribution labels with Matplotlib mathtext for plot titles."""
    if label.startswith('kappa='):
        value = label.split('=', 1)[1]
        return rf'$\kappa={value}$'
    return label
def plot_growth_maps(outdir, kpar, kperp, maps, labels):
    n = len(maps)
    ncols = 3
    nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7*ncols, 3.8*nrows), constrained_layout=True, squeeze=False)

    finite_chunks = [np.asarray(m)[np.isfinite(m)] for m in maps]
    finite_values = np.concatenate([a for a in finite_chunks if a.size])
    vlim = float(np.max(np.abs(finite_values)))
    if not np.isfinite(vlim) or vlim == 0.0:
        vlim = 1.0
    linthresh = 0.02 * vlim
    norm = SymLogNorm(linthresh=linthresh, linscale=1.0,
                      vmin=-vlim, vmax=vlim, base=10)

    levels = np.unique(np.concatenate((
        -np.geomspace(vlim, linthresh, 10),
        np.linspace(-linthresh, linthresh, 13),
        np.geomspace(linthresh, vlim, 10),
    )))

    for i, (m, label) in enumerate(zip(maps, labels)):
        ax = axes.flat[i]
        im = ax.contourf(kpar, kperp, m, levels=levels, norm=norm,
                         cmap='RdBu_r', extend='both')
        idx = np.unravel_index(np.nanargmax(m), m.shape)
        ax.plot(kpar[idx[1]], kperp[idx[0]], marker='*', linestyle='none',
                ms=10, color='black', markeredgecolor='white', label='grid maximum')
        ax.set_title(format_distribution_title(label), fontsize=14)
        ax.set_xlabel(r'$k_\parallel c/\omega_{pe}$', fontsize=13)
        ax.set_ylabel(r'$k_\perp c/\omega_{pe}$', fontsize=13)
        
        ax.legend(fontsize=7, loc='best')

    for j in range(len(maps), len(axes.flat)):
        axes.flat[j].axis('off')

    ticks = [-vlim, -0.1*vlim, -linthresh, 0.0,
             linthresh, 0.1*vlim, vlim]
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92,
                        ticks=ticks, format='%.1e')
    cbar.set_label(r'$\zeta/\omega_{pe}$', fontsize=13)
    cbar.ax.tick_params(labelsize=10)
    fig.savefig(outdir/'02_growth_rate_maps.png', dpi=800)
    plt.close(fig)
def summarize_map(growth, kpar, kperp):
    idx = np.unravel_index(np.nanargmax(growth), growth.shape)
    kz = float(kpar[idx[1]])
    kx = float(kperp[idx[0]])
    gmax = float(growth[idx])
    theta = float(np.degrees(np.arctan2(kx, kz)))
    return {'Gamma_max_over_omega_pe': gmax, 'K_parallel_max': kz,
            'K_perp_max': kx, 'theta_k_deg': theta,
            'max_is_positive': bool(gmax > 0.0)}
def plot_kappa_trends(outdir, kappas, summary_rows):
    finite = [row for row in summary_rows if str(row['distribution']).startswith('kappa=')]
    gauss = next(row for row in summary_rows if row['distribution'] == 'Gaussian')
    mj = next(row for row in summary_rows if row['distribution'] == 'Maxwell-Jüttner')
    kk = np.array([float(row['kappa']) for row in finite])
    gm = np.array([float(row['Gamma_max_over_omega_pe']) for row in finite])
    ang = np.array([float(row['theta_k_deg']) for row in finite])

    gamma_gauss = float(gauss['Gamma_max_over_omega_pe'])
    gamma_mj = float(mj['Gamma_max_over_omega_pe'])
    ratio_gauss = gm / gamma_gauss if abs(gamma_gauss) > 1e-30 else np.full_like(gm, np.nan)
    ratio_mj = gm / gamma_mj if abs(gamma_mj) > 1e-30 else np.full_like(gm, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    axes[0].semilogx(kk, gm, 'o-', label=r'$\kappa$ distribution')
    axes[0].axhline(gamma_gauss, ls='--', label='matched Gaussian')
    axes[0].set_xlabel(r'$\kappa$')
    axes[0].set_ylabel(r'$\zeta_{\max}/\omega_{pe}$')
    axes[0].set_title('Maximum weak-beam growth rate')
    axes[1].semilogx(kk, ang, 'o-', label=r'$\kappa$ distribution')
    axes[1].axhline(float(gauss['theta_k_deg']), ls='--', label='matched Gaussian')
    axes[1].set_xlabel(r'$\kappa$')
    axes[1].set_ylabel(r'$\theta_{k,\max}$ (degrees)')
    axes[1].set_title('Angle of grid-maximum mode')
    for ax in axes:
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(outdir/'03_gamma_max_and_angle_vs_kappa.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    ax.semilogx(kk, ratio_gauss, 'o-', label=r'$\zeta_{\max,\kappa}/\zeta_{\max,\mathrm{G}}$')
    ax.semilogx(kk, ratio_mj, 's--', label=r'$\zeta_{\max,\kappa}/\zeta_{\max,\mathrm{MJ}}$')
    ax.axhline(1.0, color='black', ls=':', lw=1)
    ax.set_xlabel(r'$\kappa$', fontsize=13)
    ax.set_ylabel('Growth-rate ratio', fontsize=13)
    ax.legend(fontsize=10)

    ax.grid(False)#, which='both', alpha=0.25)
    fig.savefig(outdir/'04_growth_ratio_vs_kappa.png', dpi=500)
    plt.close(fig)
def plot_rmax_convergence(outdir, params, kappa, kpar, kperp, rmax_sigmas, n_r, n_phi):
    dist = Distribution('kappa', kappa, params)
    rows = []
    vals = []
    
    for multiple in rmax_sigmas:
        rmax = float(multiple * params.sigma_perp_1d)
        
        nr_local = max(64, int(np.ceil(n_r * multiple / 20.0)))
        val = gamma_over_omega_pe(kpar, kperp, dist, rmax, n_r=nr_local, n_phi=n_phi)
        vals.append(val)
        rows.append({'r_max_over_sigma_perp': multiple, 'r_max': rmax,
                     'n_r': nr_local, 'n_phi': n_phi,
                     'Gamma_over_omega_pe': val})
    save_csv(outdir/'convergence_rmax.csv', rows,
             ['r_max_over_sigma_perp', 'r_max', 'n_r', 'n_phi', 'Gamma_over_omega_pe'])
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    ax.plot([r['r_max_over_sigma_perp'] for r in rows], vals, 'o-')
    ax.set_xlabel(r'$r_{\max}/\sigma_{\perp,1D}$')
    ax.set_ylabel(r'$\zeta/\omega_{pe}$')
    ax.set_title(fr'Radial-domain convergence ($\kappa={kappa:g}$)')
    ax.grid(True, alpha=0.25)
    fig.savefig(outdir/'05_convergence_rmax.png', dpi=220)
    plt.close(fig)
def plot_quadrature_convergence(outdir, params, kappa, kpar, kperp, rmax, quick=False):
    """Check fixed-point quadrature-order convergence for a selected kappa."""
    dist = Distribution('kappa', kappa, params)
    orders = [(32, 24), (48, 32), (72, 48), (104, 64)] if quick else [
        (48, 32), (72, 48), (104, 64), (144, 88), (200, 120), (260, 160)
    ]
    rows = []
    for nr, nphi in orders:
        val = gamma_over_omega_pe(kpar, kperp, dist, rmax, n_r=nr, n_phi=nphi)
        rows.append({'n_r': nr, 'n_phi': nphi, 'Gamma_over_omega_pe': val})
    ref = rows[-1]['Gamma_over_omega_pe']
    for row in rows:
        row['relative_difference_from_last'] = abs(row['Gamma_over_omega_pe']-ref)/max(abs(ref), 1e-30)
    save_csv(outdir/'quadrature_convergence.csv', rows,
             ['n_r', 'n_phi', 'Gamma_over_omega_pe', 'relative_difference_from_last'])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    labels = [f"{r['n_r']}x{r['n_phi']}" for r in rows]
    axes[0].plot(range(len(rows)), [r['Gamma_over_omega_pe'] for r in rows], 'o-')
    axes[0].set_xticks(range(len(rows)), labels, rotation=35, ha='right')
    axes[0].set_xlabel('Gauss–Legendre orders (n_r × n_phi)')
    axes[0].set_ylabel(r'$\zeta/\omega_{pe}$')
    axes[0].set_title('Fixed-wavevector quadrature convergence')
    axes[1].semilogy(range(len(rows)), [max(r['relative_difference_from_last'], 1e-16) for r in rows], 'o-')
    axes[1].set_xticks(range(len(rows)), labels, rotation=35, ha='right')
    axes[1].set_xlabel('Gauss–Legendre orders (n_r × n_phi)')
    axes[1].set_ylabel('Relative difference from highest order')
    axes[1].set_title('Estimated quadrature error')
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(outdir/'06_quadrature_convergence.png', dpi=220)
    plt.close(fig)
def plot_wavegrid_convergence(outdir, params, kappa, kperp_max, rmax, nkpar, nkperp, quick=False):
    """Test how the maximum changes as the wavevector grid is refined."""
    dist = Distribution('kappa', kappa, params)
    grids = [
        (max(13, nkpar//4), max(9, nkperp//3)),
        (max(21, nkpar//2), max(13, nkperp//2)),
        (nkpar, nkperp),
    
        (2 * (nkpar - 1) + 1, 2 * (nkperp - 1) + 1),
    ]
    
    grids = list(dict.fromkeys(grids))
    rows = []
    for ik, (np_k, nx_k) in enumerate(grids):
        kp, kx = scan_axes(params, np_k, nx_k, kperp_max=kperp_max)
        nr = max(64, min(120, nkpar + nkperp))
        nphi = 48 if quick else 64
        g = scan_growth_rate(dist, kp, kx, rmax, n_r=nr, n_phi=nphi)
        summ = summarize_map(g, kp, kx)
        rows.append({'nk_parallel': np_k, 'nk_perp': nx_k,
                     'n_r': nr, 'n_phi': nphi, **summ})
    ref = rows[-1]['Gamma_max_over_omega_pe']
    for row in rows:
        row['relative_difference_from_finest'] = abs(row['Gamma_max_over_omega_pe']-ref)/max(abs(ref),1e-30)
    save_csv(outdir/'wavevector_grid_convergence.csv', rows,
             ['nk_parallel','nk_perp','n_r','n_phi','Gamma_max_over_omega_pe',
              'K_parallel_max','K_perp_max','theta_k_deg','max_is_positive',
              'relative_difference_from_finest'])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    labels = [rf"${r['nk_parallel']} \times {r['nk_perp']}$"for r in rows]
    axes[0].plot(range(len(rows)), [r['Gamma_max_over_omega_pe'] for r in rows], 'o-')
    axes[0].set_xticks(range(len(rows)), labels)
    axes[0].set_xlabel(r'Grid size (parallel $\times$ transverse)', fontsize=13)
    axes[0].set_ylabel(r'$\zeta_{\max,\kappa}/\omega_{pe}$', fontsize=13)
    #axes[0].set_title('Wavevector-grid convergence')
    axes[1].semilogy(range(len(rows)), [max(r['relative_difference_from_finest'],1e-16) for r in rows], 'o-')
    axes[1].set_xticks(range(len(rows)), labels)
    axes[1].set_xlabel(r'Grid size (parallel $\times$ transverse)', fontsize=13)
    axes[1].set_ylabel('Relative difference from finest grid', fontsize=13)
    #axes[1].set_title('Maximum-growth convergence')
    for ax in axes:
        ax.grid(False)#, alpha=0.25)
        ax.text(0.06, 0.95, rf'$\kappa = {kappa:g}$', transform=ax.transAxes, ha='left',
        va='top',
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )
    fig.savefig(outdir/'07_wavevector_grid_convergence.png', dpi=800)
    plt.close(fig)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=Path('kappa_results'))
    parser.add_argument('--alpha', type=float, default=1e-9, help='n_b/n_p; illustrative default')
    parser.add_argument('--p0', type=float, default=10.0, help='p0/(m_e c)')
    parser.add_argument('--sigma-parallel', type=float, default=0.5, help='1-D rms parallel momentum width /(m_e c)')
    parser.add_argument('--sigma-perp', type=float, default=0.1, help='1-D rms transverse-component width /(m_e c)')
    parser.add_argument('--kappas', type=float, nargs='+', default=[1.7, 2, 3, 5, 10, 30, 100])
    parser.add_argument('--n-r', type=int, default=180)
    parser.add_argument('--n-phi', type=int, default=80)
    parser.add_argument('--nkpar', type=int, default=81)
    parser.add_argument('--nkperp', type=int, default=41)
    parser.add_argument('--kperp-max', type=float, default=3.0, help='maximum scanned K_perp; use broad enough range to test oblique modes')
    parser.add_argument('--r-max-sigmas', type=float, default=20.0)
    parser.add_argument('--quick', action='store_true', help='reduce grid sizes for a fast smoke test')
    args = parser.parse_args()
    if args.quick:
        args.n_r = min(args.n_r, 80)
        args.n_phi = min(args.n_phi, 48)
        args.nkpar = min(args.nkpar, 31)
        args.nkperp = min(args.nkperp, 17)
    args.outdir.mkdir(parents=True, exist_ok=True)
    params = BeamParameters(args.alpha, args.p0, args.sigma_parallel, args.sigma_perp)
    dists = make_dists(params, args.kappas)
    mj_dist = next(dist for label, dist in dists if label == 'Maxwell-Jüttner')
    print(
        'Maxwell-Jüttner fit: ' 
        f'theta={mj_dist.mj_theta:.6g}, beta_d={mj_dist.mj_drift_beta:.6g}, '
        f'predicted sigma_parallel={mj_dist.mj_sigma_parallel:.6g} m_e c '
        f'(requested kappa/Gaussian width={params.sigma_parallel:.6g} m_e c).'
    )
    kpar, kperp = scan_axes(params, args.nkpar, args.nkperp, kperp_max=args.kperp_max)
    rmax = args.r_max_sigmas * params.sigma_perp_1d
    config = {
        'warning': 'Illustrative defaults; replace with physically justified beam parameters before publication.',
        'alpha_nb_over_np': params.alpha, 'p0_over_mec': params.p0,
        'sigma_parallel_1d_over_mec': params.sigma_parallel,
        'sigma_perp_1d_over_mec': params.sigma_perp_1d,
        'kappas': args.kappas, 'Gaussian included': True, 'Maxwell-Juttner included': True,
        'r_max_over_mec': rmax, 'n_r': args.n_r, 'n_phi': args.n_phi,
        'nkpar': args.nkpar, 'nkperp': args.nkperp,
        'kpar_min': float(kpar.min()), 'kpar_max': float(kpar.max()),
        'kperp_min': float(kperp.min()), 'kperp_max': float(kperp.max()),
        'model': 'weak-beam electrostatic Langmuir branch; cold background; K_parallel>1'
    }
    mj_dist = next(dist for label, dist in dists if label == 'Maxwell-Jüttner')
    config['maxwell_juttner_fit'] = {
        'dimensionless_temperature_kT_over_mec2': mj_dist.mj_theta,
        'drift_beta': mj_dist.mj_drift_beta,
        'drift_gamma': mj_dist.mj_drift_gamma,
        'predicted_sigma_parallel_1d_over_mec': mj_dist.mj_sigma_parallel,
        'fitted_to': 'mean p_parallel = p0 and transverse 1D RMS sigma_perp',
        'note': 'The longitudinal RMS width is predicted and need not equal sigma_parallel.'
    }
    (args.outdir/'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Computing matched distribution marginals...')
    plot_marginals(args.outdir, params, dists)
    map_dists = dists
    maps, labels, rows = [], [], []
    for label, dist in map_dists:
        print(f'Scanning {label}: nr={args.n_r}, nphi={args.n_phi}, grid={len(kperp)}x{len(kpar)}')
        growth = scan_growth_rate(dist, kpar, kperp, rmax, n_r=args.n_r, n_phi=args.n_phi)
        maps.append(growth)
        labels.append(label)
        summary = summarize_map(growth, kpar, kperp)
        summary['distribution'] = label
        summary['kappa'] = float(dist.kappa) if dist.kind == 'kappa' else ''
        rows.append(summary)
        safe_label = label.replace('=', '').replace('.', 'p')
        np.savetxt(args.outdir/f'growth_map_{safe_label}.csv', growth, delimiter=',',
                   header='rows=k_perp increasing; columns=k_parallel increasing', comments='')
        print('  ', summary)
    plot_growth_maps(args.outdir, kpar, kperp, maps, labels)
    save_csv(args.outdir/'growth_maxima.csv', rows,
             ['distribution', 'kappa', 'Gamma_max_over_omega_pe', 'K_parallel_max',
              'K_perp_max', 'theta_k_deg', 'max_is_positive'])
    plot_kappa_trends(args.outdir, args.kappas, rows)
    kcrit = params.k_parallel_resonant_centre + params.dk_parallel_per_sigma
    kx_conv = 3.0*params.sigma_perp_1d/params.p0
    print('Computing radial truncation convergence at one selected wavevector...')
    kappa_check = float(args.kappas[1] if len(args.kappas)>1 else args.kappas[0])
    plot_rmax_convergence(args.outdir, params, kappa_check,
                          kcrit, kx_conv, [8, 12, 16, 20, 28],
                          max(args.n_r, 100), max(args.n_phi, 64))
    print('Computing fixed-point quadrature-order convergence...')
    plot_quadrature_convergence(args.outdir, params, kappa_check, kcrit, kx_conv, rmax, quick=args.quick)
    print('Computing wavevector-grid convergence...')
    plot_wavegrid_convergence(args.outdir, params, kappa_check, args.kperp_max, rmax,
                              args.nkpar, args.nkperp, quick=args.quick)
    print(f'\nDone. Results written to: {args.outdir.resolve()}')
    print('Figures:')
    for path in sorted(args.outdir.glob('*.png')):
        print(' ', path.name)
    print('\nReminder: these outputs are numerical results for the illustrative parameter defaults, not paper-ready physical conclusions.')
if __name__ == '__main__':
    main()
