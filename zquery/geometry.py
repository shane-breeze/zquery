import numpy as np
import numba as nb

__all__ = [
    "bound_phi",
    "deltar",
    "radial_to_cartesian2d",
    "cartesian_to_radial2d",
    "particle_to_cartesian3d",
    "cartesian_to_particle3d",
    "particle_to_lorentz4d",
    "lorentz_to_particle4d",
]

@nb.njit
def bound_phi(phi):
    mask = (phi >= np.pi)
    while np.any(mask):
        phi[mask] = phi[mask] - 2*np.pi
        mask = (phi >= np.pi)

    mask = (phi < -np.pi)
    while np.any(mask):
        phi[mask] = phi[mask] + 2*np.pi
        mask = (phi < -np.pi)

    return phi

@nb.njit
def deltar(deta, dphi):
    return np.sqrt(deta**2 + bound_phi(dphi)**2)

@nb.njit
def radial_to_cartesian2d(pt, phi):
    return pt*np.cos(phi), pt*np.sin(phi)

@nb.njit
def cartesian_to_radial2d(px, py):
    return np.sqrt(px**2 + py**2), bound_phi(np.arctan2(py, px))

@nb.njit
def particle_to_cartesian3d(pt, eta, phi):
    px, py = radial_to_cartesian2d(pt, phi)
    return px, py, pt*np.sinh(eta)

@nb.njit
def cartesian_to_particle3d(px, py, pz):
    pt, phi = cartesian_to_radial2d(px, py)
    return pt, np.arctanh(pz/np.sqrt(pz**2 + pt**2)), phi

@nb.njit
def particle_to_lorentz4d(pt, eta, phi, mass):
    px, py, pz = particle_to_cartesian3d(pt, eta, phi)
    return px, py, pz, np.sqrt(mass**2 + pt**2 + pz**2)

@nb.njit
def lorentz_to_particle4d(px, py, pz, en):
    pt, eta, phi = cartesian_to_particle3d(px, py, pz)
    mass2 = en**2 - pt**2 - pz**2
    mass = np.sign(mass2)*np.sqrt(np.abs(mass2))
    return pt, eta, phi, mass
