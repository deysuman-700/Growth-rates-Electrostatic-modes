#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Optional
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaln, kve


@lru_cache(maxsize=256)
def _mj_rest_q2_mean(theta: float) -> float:
    
    theta = float(theta)
    if theta <= 0.0:
        raise ValueError("Maxwell-Juttner temperature theta must be positive")
    scale = np.sqrt(theta * (1.0 + theta))

    def weight(u):
        q = scale * u
        gamma_minus_one = q*q / (np.sqrt(1.0 + q*q) + 1.0)
        return np.exp(-gamma_minus_one / theta)

    i2 = quad(lambda u: u*u*weight(u), 0.0, np.inf, epsabs=1e-11,
              epsrel=2e-9, limit=250)[0]
    i4 = quad(lambda u: u**4*weight(u), 0.0, np.inf, epsabs=1e-11,
              epsrel=2e-9, limit=250)[0]
    return float(scale*scale*i4/i2)


@dataclass(frozen=True)
class BeamParameters:

    alpha: float = 1.0e-9          # n_b / n_p
    p0: float = 10.0               # mean parallel momentum / (m_e c)
    sigma_parallel: float = 0.5    # rms p_parallel width / (m_e c)
    sigma_perp_1d: float = 0.1     # rms p_x or p_y width / (m_e c)

    def __post_init__(self):
        if self.alpha <= 0:
            raise ValueError("alpha=n_b/n_p must be positive")
        if self.p0 <= 0:
            raise ValueError("p0 must be positive")
        if self.sigma_parallel <= 0 or self.sigma_perp_1d <= 0:
            raise ValueError("Both rms momentum widths must be positive")

    @property
    def beta0(self) -> float:
        return self.p0 / np.sqrt(1.0 + self.p0**2)

    @property
    def k_parallel_resonant_centre(self) -> float:
        
        return 1.0 / self.beta0

    @property
    def dk_parallel_per_sigma(self) -> float:
        
        return self.sigma_parallel / (self.p0**2 * np.sqrt(1.0 + self.p0**2))


@dataclass(frozen=True)
class Distribution:
    
    kind: str
    kappa: Optional[float]
    params: BeamParameters

    def __post_init__(self):
        if self.kind not in ("kappa", "gaussian", "maxwell_juttner"):
            raise ValueError("kind must be 'kappa', 'gaussian', or 'maxwell_juttner'")
        if self.kind == "kappa" and (self.kappa is None or self.kappa <= 1.5):
            raise ValueError("Finite matched rms widths require kappa > 3/2")

    @cached_property
    def mj_theta(self) -> float:
        
        target = self.params.sigma_perp_1d
        def residual(log_theta):
            th = float(np.exp(log_theta))
            return np.sqrt(_mj_rest_q2_mean(th) / 3.0) - target

        lo = np.log(max(1e-7, target*target*1e-5))
        hi = np.log(max(0.1, 10.0*target*target, 2.0*target))
        while residual(hi) < 0.0:
            hi += np.log(10.0)
            if hi > np.log(1e6):
                raise RuntimeError("Could not fit Maxwell-Juttner temperature to transverse RMS width")
        return float(np.exp(brentq(residual, lo, hi, xtol=1e-11, rtol=1e-10)))

    @cached_property
    def mj_drift_gamma(self) -> float:
        
        th = self.mj_theta
        x = 1.0 / th
        mean_gamma_rest = float(kve(1, x) / kve(2, x) + 3.0*th)
        enthalpy_per_particle = mean_gamma_rest + th
        drift_four_velocity = self.params.p0 / enthalpy_per_particle
        return float(np.sqrt(1.0 + drift_four_velocity**2))

    @cached_property
    def mj_drift_beta(self) -> float:
        g = self.mj_drift_gamma
        return float(np.sqrt(1.0 - 1.0/(g*g)))

    @property
    def mj_sigma_parallel(self) -> float:
        
        th = self.mj_theta
        q2 = _mj_rest_q2_mean(th)
        gd = self.mj_drift_gamma
        beta = self.mj_drift_beta
        second = gd**2 * (q2/3.0 + beta**2*(1.0 + 5.0*q2/3.0))
        variance = max(0.0, second - self.params.p0**2)
        return float(np.sqrt(variance))

    @property
    def a_parallel(self) -> float:
        if self.kind == "gaussian":
            return np.sqrt(2.0) * self.params.sigma_parallel
        if self.kind == "maxwell_juttner":
            return self.mj_sigma_parallel
        k = float(self.kappa)
        return self.params.sigma_parallel * np.sqrt((2.0 * k - 3.0) / k)

    @property
    def a_perp(self) -> float:
        if self.kind == "gaussian":
            return np.sqrt(2.0) * self.params.sigma_perp_1d
        if self.kind == "maxwell_juttner":
            return self.params.sigma_perp_1d
        k = float(self.kappa)
        return self.params.sigma_perp_1d * np.sqrt((2.0 * k - 3.0) / k)

    @property
    def normalization(self) -> float:
        
        if self.kind == "maxwell_juttner":
            th = self.mj_theta
            x = 1.0/th
            logc = x - np.log(4.0*np.pi*self.mj_drift_gamma*th*kve(2, x))
            return float(np.exp(logc)) if logc < 700.0 else float("inf")
        ap, az = self.a_perp, self.a_parallel
        if self.kind == "gaussian":
            return 1.0 / (np.pi**1.5 * ap**2 * az)
        k = float(self.kappa)
        logc = (gammaln(k + 1.0) - 1.5 * np.log(np.pi)
                - 1.5 * np.log(k) - 2.0 * np.log(ap) - np.log(az)
                - gammaln(k - 0.5))
        return float(np.exp(logc))


def distribution_and_derivatives(r, z, dist: Distribution):
    
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    p = dist.params

    if dist.kind == "maxwell_juttner":
        th = dist.mj_theta
        beta_d = dist.mj_drift_beta
        gamma_d = dist.mj_drift_gamma
        particle_gamma = np.sqrt(1.0 + r*r + z*z)
        
        x = 1.0 / th
        energy_comoving = gamma_d * (particle_gamma - beta_d*z)
        logf = -(energy_comoving - 1.0)/th - np.log(4.0*np.pi*gamma_d*th*kve(2, x))
        F = np.exp(logf)
        Fr = -gamma_d*r/(th*particle_gamma)*F
        Fz = -gamma_d*(z/particle_gamma - beta_d)/th*F
        return F, Fr, Fz

    ap, az = dist.a_perp, dist.a_parallel
    dz = z - p.p0
    c0 = dist.normalization
    if dist.kind == "gaussian":
        q = (r / ap)**2 + (dz / az)**2
        F = c0 * np.exp(-q)
        Fr = -2.0 * r / ap**2 * F
        Fz = -2.0 * dz / az**2 * F
    else:
        k = float(dist.kappa)
        q = 1.0 + ((r / ap)**2 + (dz / az)**2) / k
        F = c0 * np.exp(-(k + 1.0) * np.log(q))
        common = -2.0 * (k + 1.0) / (k * q) * F
        Fr = common * r / ap**2
        Fz = common * dz / az**2
    return F, Fr, Fz


def _mj_quadrature_limits(dist: Distribution):
    
    p = dist.params
    rmax = max(1.5, 14.0*p.sigma_perp_1d)
    zwidth = max(12.0*dist.mj_sigma_parallel, 12.0*p.sigma_parallel, 2.0)
    return rmax, p.p0-zwidth, p.p0+zwidth


def parallel_marginal(y, dist: Distribution):
    
    p = dist.params
    y = np.asarray(y, dtype=float)
    z = p.p0 + y*p.sigma_parallel
    if dist.kind == "maxwell_juttner":
        rmax, _, _ = _mj_quadrature_limits(dist)
        r, wr = _gauss_legendre_interval(180, 0.0, rmax)
        R = r[:, None]
        Z = z[None, :]
        F, _, _ = distribution_and_derivatives(R, Z, dist)
        pdf = 2.0*np.pi*np.sum((wr*r)[:, None]*F, axis=0)
        return p.sigma_parallel*pdf

    a = dist.a_parallel
    x = y*p.sigma_parallel
    if dist.kind == "gaussian":
        pdf_x = np.exp(-(x/a)**2) / (np.sqrt(np.pi)*a)
    else:
        k = float(dist.kappa)
        log_pref = (gammaln(k) - 0.5*np.log(np.pi*k) - np.log(a)
                    - gammaln(k - 0.5))
        pdf_x = np.exp(log_pref - k*np.log1p((x/a)**2/k))
    return p.sigma_parallel*pdf_x


def transverse_component_marginal(y, dist: Distribution):
    
    p = dist.params
    y = np.asarray(y, dtype=float)
    xvals = y*p.sigma_perp_1d
    if dist.kind == "maxwell_juttner":
        rmax, zlo, zhi = _mj_quadrature_limits(dist)
        py, wpy = _gauss_legendre_interval(160, -rmax, rmax)
        z, wz = _gauss_legendre_interval(200, zlo, zhi)
        result = np.empty_like(xvals)
        
        for begin in range(0, xvals.size, 24):
            end = min(begin+24, xvals.size)
            xx = xvals[begin:end, None, None]
            R = np.sqrt(xx*xx + py[None, :, None]**2)
            Z = z[None, None, :]
            F, _, _ = distribution_and_derivatives(R, Z, dist)
            result[begin:end] = np.sum(
                F*wpy[None, :, None]*wz[None, None, :], axis=(1, 2)
            )
        return p.sigma_perp_1d*result

    a = dist.a_perp
    if dist.kind == "gaussian":
        pdf_x = np.exp(-(xvals/a)**2) / (np.sqrt(np.pi)*a)
    else:
        k = float(dist.kappa)
        log_pref = (gammaln(k) - 0.5*np.log(np.pi*k) - np.log(a)
                    - gammaln(k - 0.5))
        pdf_x = np.exp(log_pref - k*np.log1p((xvals/a)**2/k))
    return p.sigma_perp_1d*pdf_x


def _gauss_legendre_interval(n: int, lo: float, hi: float):
    nodes, weights = leggauss(int(n))
    return 0.5*(hi-lo)*nodes + 0.5*(hi+lo), 0.5*(hi-lo)*weights


def resonant_integral_reduced(
    k_parallel: float,
    k_perp: float,
    dist: Distribution,
    r_max: float,
    n_r: int = 160,
    n_phi: int = 96,
) -> float:
    
    Kz, Kx = float(k_parallel), float(k_perp)
    if Kz <= 1.0:
        raise ValueError("This resonance-surface parameterization requires K_parallel > 1")
    if Kx < 0 or r_max <= 0:
        raise ValueError("k_perp must be nonnegative and r_max positive")

    r, wr = _gauss_legendre_interval(n_r, 0.0, r_max)
    phi, wphi = _gauss_legendre_interval(n_phi, 0.0, np.pi)
    R = r[:, None]
    PH = phi[None, :]
    W = (wr[:, None] * wphi[None, :])
    rho = np.cos(PH)
    b = Kx * R * rho
    denom = Kz**2 - 1.0
    disc = np.sqrt(b**2 + denom * (1.0 + R**2))
    Z = (-Kz*b + disc) / denom
    gamma = np.sqrt(1.0 + R**2 + Z**2)
    beta_z = Z / gamma
    _, Fr, Fz = distribution_and_derivatives(R, Z, dist)
    denominator = Kz - beta_z
    if np.any(denominator <= 0.0):
        raise FloatingPointError("Nonpositive K_parallel-beta_parallel encountered")
    integrand = 2.0 * R * gamma * (Kz*Fz + Kx*rho*Fr) / denominator
    return float(np.sum(W * integrand))


def resonant_integral_from_original_3d(
    k_parallel: float,
    k_perp: float,
    dist: Distribution,
    r_max: float,
    n_r: int = 160,
    n_phi: int = 128,
) -> float:
    
    Kz, Kx = float(k_parallel), float(k_perp)
    if Kz <= 1.0:
        raise ValueError("This resonance-surface parameterization requires K_parallel > 1")
    if Kx < 0 or r_max <= 0:
        raise ValueError("k_perp must be nonnegative and r_max positive")

    r, wr = _gauss_legendre_interval(n_r, 0.0, r_max)
    phi, wphi = _gauss_legendre_interval(n_phi, 0.0, 2.0*np.pi)
    R = r[:, None]
    PH = phi[None, :]
    W = wr[:, None] * wphi[None, :]
    cosphi = np.cos(PH)
    q = Kx * R * cosphi
    denom_quad = Kz**2 - 1.0
    disc = np.sqrt(q**2 + denom_quad * (1.0 + R**2))
    Z = (-Kz*q + disc) / denom_quad
    gamma = np.sqrt(1.0 + R**2 + Z**2)
    _, Fr, Fz = distribution_and_derivatives(R, Z, dist)
    numerator = Kz*(1.0 + R**2) - q*Z
    dh_dz = numerator / gamma**3
    if np.any(np.abs(dh_dz) < 1e-14):
        raise FloatingPointError("Resonance Jacobian is too close to zero")
    integrand = R * (Kz*Fz + Kx*cosphi*Fr) / np.abs(dh_dz)
    return float(np.sum(W * integrand))


def gamma_over_omega_pe(
    k_parallel: float,
    k_perp: float,
    dist: Distribution,
    r_max: float,
    n_r: int = 160,
    n_phi: int = 96,
    method: str = "reduced",
) -> float:
    
    K2 = float(k_parallel)**2 + float(k_perp)**2
    if K2 <= 0:
        raise ValueError("Wavevector magnitude must be positive")
    if method == "reduced":
        J = resonant_integral_reduced(k_parallel, k_perp, dist, r_max, n_r, n_phi)
    elif method == "original3d":
        J = resonant_integral_from_original_3d(
            k_parallel, k_perp, dist, r_max, n_r, max(n_phi, 2*n_phi//3)
        )
    else:
        raise ValueError("method must be 'reduced' or 'original3d'")
    return float(0.5*np.pi * dist.params.alpha * J / K2)


def scan_growth_rate(
    dist: Distribution,
    k_parallel_values: np.ndarray,
    k_perp_values: np.ndarray,
    r_max: float,
    n_r: int = 120,
    n_phi: int = 72,
):
    
    gamma = np.empty((len(k_perp_values), len(k_parallel_values)), dtype=float)
    for i, kx in enumerate(k_perp_values):
        for j, kz in enumerate(k_parallel_values):
            gamma[i, j] = gamma_over_omega_pe(
                kz, kx, dist, r_max, n_r=n_r, n_phi=n_phi, method="reduced"
            )
    return gamma


def normalized_moment_checks(dist: Distribution) -> dict:
    
    out = {
        "density_target": 1.0,
        "mean_parallel_target": dist.params.p0,
        "sigma_parallel_target": dist.params.sigma_parallel,
        "sigma_perp_1d_target": dist.params.sigma_perp_1d,
        "normalization_C": dist.normalization,
    }
    if dist.kind == "maxwell_juttner":
        out.update({
            "theta_kT_over_mec2": dist.mj_theta,
            "drift_beta": dist.mj_drift_beta,
            "drift_gamma": dist.mj_drift_gamma,
            "sigma_parallel_predicted": dist.mj_sigma_parallel,
            "sigma_perp_1d_predicted": dist.params.sigma_perp_1d,
        })
    else:
        out.update({"a_parallel": dist.a_parallel, "a_perp": dist.a_perp})
    return out
