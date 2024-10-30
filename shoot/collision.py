import numpy as np

from .math import unit, orth


class AARect:
    """Axis-aligned rectangle."""

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.vertices = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
        self.normals = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        self.segments = [
            Segment(self.vertices[i - 1], self.vertices[i]) for i in range(4)
        ]


class Segment:
    """Line segment."""

    def __init__(self, s1, s2):
        self.s1 = np.array(s1)
        self.s2 = np.array(s2)

        self.v = self.s2 - self.s1
        self.normal = unit(orth(self.v))


class Circle:
    def __init__(self, c, r):
        self.c = c
        self.r = r


def line_rect_edge_intersection(p, v, rect):
    """Compute the intersection of a line with the edge of the screen.

    Parameters
    ----------
    p : pair of float
        Starting point of the line.
    v : pair of float
        Direction vector of the line (does not need to be normalized).
    rect : AARect
        Rectangle of the screen.

    Returns
    -------
    : pair of float
        The closest intersection point corresponding to non-negative distance
        along direction v.
    """
    # TODO can generalize to get rid of this
    assert rect.x == 0
    assert rect.y == 0

    ts = []

    # vertical edges
    if not np.isclose(v[0], 0):
        ts.extend([-p[0] / v[0], (rect.w - p[0]) / v[0]])

    # horizontal edges
    if not np.isclose(v[1], 0):
        ts.extend([-p[1] / v[1], (rect.h - p[1]) / v[1]])

    # return the smallest positive value
    ts = np.array(ts)
    t = np.min(ts[ts >= 0])
    return p + t * v




def point_in_rect(point, rect):
    """True if the rectangle contains the point, False otherwise."""
    x, y = point
    return x >= rect.x and x <= rect.x + rect.w and y >= rect.y and y <= rect.y + rect.h


def point_segment_dist(point, segment):
    """Distance between a point and a line segment."""
    q = np.array(segment.s1 - point)

    t = -(q @ segment.v) / (segment.v @ segment.v)
    if t >= 0 and t <= 1:
        r = segment.s1 + t * segment.v
        return np.linalg.norm(point - r)
    d1 = np.linalg.norm(point - segment.s1)
    d2 = np.linalg.norm(point - segment.s2)
    return min(d1, d2)


def circle_segment_intersect(circle, segment):
    """True if circle and line segment intersect, False otherwise."""
    return point_segment_dist(circle.c, segment) <= circle.r


def _quad_formula(a, b, c):
    d = np.sqrt(b**2 - 4 * a * c)
    return (-b - d) / (2 * a), (-b + d) / (2 * a)


def circle_segment_intersect_dist(circle, segment):
    # the segment starts in the circle already
    if np.linalg.norm(segment.s1 - circle.c) <= circle.r:
        return 0

    q = segment.s1 - circle.c
    v = segment.v

    a = v @ v
    b = 2 * q @ v
    c = q @ q - circle.r**2
    ts = _quad_formula(a, b, c)
    return np.min(ts)


def segment_segment_dist(segment1, segment2):
    """Distance between two line segments."""
    # segments are parallel
    if np.isclose(segment1.normal @ segment2.v, 0):
        return np.abs(segment1.normal @ (segment2.s1 - segment1.s1))

    v1 = segment1.v
    v2 = segment2.v
    d = segment1.s1 - segment2.s1

    # determine the intersection point for the infinite lines
    A = np.array([[-v1 @ v1, v1 @ v2], [-v1 @ v2, v2 @ v2]])
    b = np.array([v1 @ d, v2 @ d])
    t = np.linalg.solve(A, b)

    c1 = 0 <= t[0] <= 1
    c2 = 0 <= t[1] <= 1

    # line segments actually intersect
    if c1 and c2:
        return 0

    # intersection is outside segment2 but not segment1: closest point must be
    # an endpoint of segment2
    elif c1 and not c2:
        if t[1] > 1:
            return point_segment_dist(segment2.s2, segment1)
        return point_segment_dist(segment2.s1, segment1)

    # opposite of above
    elif c2 and not c1:
        if t[0] > 1:
            return point_segment_dist(segment1.s2, segment2)
        return point_segment_dist(segment1.s1, segment2)

    # otherwise the closest distance is between a pair of endpoints
    deltas = [
        segment1.s1 - segment2.s1,
        segment1.s1 - segment2.s2,
        segment1.s2 - segment2.s1,
        segment1.s2 - segment2.s2,
    ]
    return np.min([np.linalg.norm(delta) for delta in deltas])


def segment_segment_intersect_dist(segment1, segment2):
    # parallel
    if np.isclose(segment1.normal @ segment2.v, 0):
        return None

    v1 = segment1.v
    v2 = segment2.v
    d = segment1.s1 - segment2.s1

    # determine the intersection point for the infinite lines
    A = np.array([[-v1 @ v1, v1 @ v2], [-v1 @ v2, v2 @ v2]])
    b = np.array([v1 @ d, v2 @ d])
    t = np.linalg.solve(A, b)

    c1 = 0 <= t[0] <= 1
    c2 = 0 <= t[1] <= 1
    if c1 and c2:
        return t[0]
    return None


def segment_rect_intersect(segment, rect):
    """True if segment and rectangle are intersecting, False otherwise."""
    # look for separating axis
    normals = np.vstack((rect.normals, segment.normal))
    for normal in normals:
        s = [normal @ segment.s1, normal @ segment.s2]
        r = rect.vertices @ normal
        if np.max(s) < np.min(r) or np.min(s) > np.max(r):
            return False
    return True


def segment_rect_intersect_dist(segment, rect):
    if point_in_rect(segment.s1, rect):
        return 0
    min_dist = np.inf
    for seg in rect.segments:
        d = segment_segment_intersect_dist(segment, seg)
        if d is not None:
            min_dist = min(d, min_dist)
    return min_dist
