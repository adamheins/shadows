import numpy as np


def unit(v):
    """Normalize to a unit vector."""
    norm = np.linalg.norm(v)
    if np.isclose(norm, 0):
        return np.zeros_like(v)
    return v / norm


def orth(v):
    """Generate a 2D orthogonal to v."""
    return np.array([v[1], -v[0]])


def rotmat(angle):
    """2D rotation matrix."""
    s = np.sin(angle)
    c = np.cos(angle)
    return np.array([[c, s], [-s, c]])



