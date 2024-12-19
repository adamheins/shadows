import math
import numpy as np

ORTHMAT = np.array([[0, 1], [-1, 0]])

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
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, s], [-s, c]])


def wrap_to_pi(x):
    """Wrap a value to [-pi, pi]"""
    return math.remainder(x, 2 * np.pi)


def angle2pi(v, start=0):
    """Compute angle of vector v w.r.t. the start in the interval [0, 2pi]."""
    # negative for y is because we are in a left-handed frame
    a = np.arctan2(-v[1], v[0]) - start
    if a < 0:
        a = 2 * np.pi + a
    return a


def quad_formula(a, b, c):
    """Evaluate the quadratic formula for coefficients a, b, c.

    The two solutions are returned.
    """
    d = np.sqrt(b ** 2 - 4 * a * c)
    return (-b - d) / (2 * a), (-b + d) / (2 * a)

