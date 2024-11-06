import numpy as np

from .math import unit, orth, quad_formula


# TODO bring this back again?
class CollisionQuery:
    def __init__(
        self, distance=None, time=None, normal=None, point=None, intersect=False
    ):
        self.point = point
        self.normal = normal
        self.distance = distance
        self.time = time
        self.intersect = intersect


# TODO
class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices

        self.edges = [
            Segment(self.vertices[i], self.vertices[i + 1])
            for i in range(len(vertices) - 1)
        ]
        self.edges.append(Segment(self.vertices[-1], self.vertices[0]))

        self.in_normals = np.array([edge.normal for edge in self.edges])

    @property
    def out_normals(self):
        return -self.in_normals


class AARect(Polygon):
    """Axis-aligned rectangle.

    Parameters
    ----------
    x : float
        x-coordinate of the top-left corner.
    y : float
        y-coordinate of the top-left corner.
    w : float
        Width of the rectangle.
    h : float
        Height of the rectangle.
    """

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        vertices = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
        super().__init__(vertices)


class Segment:
    """Line segment."""

    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)

        self.v = self.end - self.start
        self.direction = unit(self.v)
        self.normal = unit(orth(self.v))


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius


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


def point_in_poly(point, poly, tol=1e-8):
    """True if the polygon contains the point, False otherwise."""
    for v, n in zip(poly.vertices, poly.in_normals):
        if (point - v) @ n < -tol:
            return CollisionQuery(intersect=False)
    return CollisionQuery(intersect=True)


def point_in_rect(point, rect):
    """Specialized version for rectangles."""
    x, y = point
    intersect = (
        x >= rect.x and x <= rect.x + rect.w and y >= rect.y and y <= rect.y + rect.h
    )
    return CollisionQuery(intersect=intersect)


def point_segment_dist(point, segment):
    """Distance between a point and a line segment."""
    q = np.array(segment.start - point)

    t = -(q @ segment.v) / (segment.v @ segment.v)
    if t >= 0 and t <= 1:
        r = segment.start + t * segment.v
        return np.linalg.norm(point - r)
    d1 = np.linalg.norm(point - segment.start)
    d2 = np.linalg.norm(point - segment.end)
    return min(d1, d2)


def segment_circle_intersect(segment, circle):
    """True if circle and line segment intersect, False otherwise."""
    return point_segment_dist(circle.center, segment) <= circle.radius


def segment_circle_intersect_time(segment, circle):
    # the segment starts in the circle already
    if np.linalg.norm(segment.start - circle.center) <= circle.radius:
        return 0

    q = segment.start - circle.center
    v = segment.v

    a = v @ v
    b = 2 * q @ v
    c = q @ q - circle.radius**2
    ts = quad_formula(a, b, c)
    return np.min(ts)


def segment_segment_dist(segment1, segment2):
    """Distance between two line segments."""
    # TODO I think this function could be simplified
    # Q = CollisionQuery()

    # segments are parallel
    if np.isclose(segment1.normal @ segment2.v, 0):
        separation = np.abs(segment1.normal @ (segment2.start - segment1.start))

        # if one of the end points of the second segment lies in the first,
        # then the distance is just the separation between them
        direction = segment1.direction
        p0 = segment1.start
        if 0 <= (segment2.start - p0) @ direction <= segment1.v @ direction:
            return separation
        if 0 <= (segment2.end - p0) @ direction <= segment1.v @ direction:
            return separation

        # otherwise the closest distance is between a pair of endpoints
        deltas = [
            segment1.start - segment2.start,
            segment1.start - segment2.end,
            segment1.end - segment2.start,
            segment1.end - segment2.end,
        ]
        return np.min([np.linalg.norm(delta) for delta in deltas])

    v1 = segment1.v
    v2 = segment2.v
    d = segment1.start - segment2.start

    # determine the intersection point for the infinite lines
    A = np.array([[-v1 @ v1, v1 @ v2], [-v1 @ v2, v2 @ v2]])
    b = np.array([v1 @ d, v2 @ d])
    t = np.linalg.solve(A, b)

    c1 = 0 <= t[0] <= 1
    c2 = 0 <= t[1] <= 1

    # line segments actually intersect
    if c1 and c2:
        return 0
        # Q.d = 0
        # Q.intersect = True
        # Q.t = t[0]
        # Q.p = segment1.start + t[0] * segment1.v
        # return Q

    # intersection is outside segment2 but not segment1: closest point must be
    # an endpoint of segment2
    elif c1 and not c2:
        if t[1] > 1:
            return point_segment_dist(segment2.end, segment1)
        return point_segment_dist(segment2.start, segment1)

    # opposite of above
    elif c2 and not c1:
        if t[0] > 1:
            return point_segment_dist(segment1.end, segment2)
        return point_segment_dist(segment1.start, segment2)

    # otherwise the closest distance is between a pair of endpoints
    deltas = [
        segment1.start - segment2.start,
        segment1.start - segment2.end,
        segment1.end - segment2.start,
        segment1.end - segment2.end,
    ]
    return np.min([np.linalg.norm(delta) for delta in deltas])


def segment_segment_intersect_time(segment1, segment2, tol=1e-8):
    # parallel
    if np.isclose(segment1.normal @ segment2.v, 0):

        # check if they live along the same line
        # if so, they could still intersect
        separation = segment1.normal @ (segment2.start - segment1.start)
        if not abs(separation) < tol:
            return None

        direction = segment1.direction
        p0 = segment1.start
        t1 = ((segment2.start - p0) @ direction) / (segment1.v @ direction)
        t2 = ((segment2.end - p0) @ direction) / (segment1.v @ direction)

        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            return min(t1, t2)
        elif 0 <= t1 <= 1:
            return t1
        elif 0 <= t2 <= 1:
            return t2
        return None

    v1 = segment1.v
    v2 = segment2.v
    d = segment1.start - segment2.start

    # determine the intersection point for the infinite lines
    A = np.array([[-v1 @ v1, v1 @ v2], [-v1 @ v2, v2 @ v2]])
    b = np.array([v1 @ d, v2 @ d])
    t = np.linalg.solve(A, b)

    c1 = 0 <= t[0] <= 1
    c2 = 0 <= t[1] <= 1
    if c1 and c2:
        return t[0]
    return None


def segment_poly_intersect(segment, poly):
    """True if segment and rectangle are intersecting, False otherwise."""
    # look for separating axis
    normals = np.vstack((poly.out_normals, segment.normal))
    for normal in normals:
        s = [normal @ segment.start, normal @ segment.end]
        r = poly.vertices @ normal
        if np.max(s) < np.min(r) or np.min(s) > np.max(r):
            return False
    return True


def segment_poly_intersect_time(segment, poly):
    """Distance along the segment at which it intersects with the rectangle.

    Returns None if the segment and rectangle are not intersecting.
    """
    if point_in_poly(segment.start, poly).intersect:
        return 0

    min_time = None
    for edge in poly.edges:
        t = segment_segment_intersect_time(segment, edge)
        if t is not None:
            if min_time is None:
                min_time = t
            else:
                min_time = min(t, min_time)
    return min_time


def point_poly_dist(point, poly):
    """Minimum distance between a point and a polygon."""
    if point_in_poly(point, poly).intersect:
        return 0

    min_dist = np.inf
    for edge in poly.edges:
        d = point_segment_dist(point, edge)
        min_dist = min(d, min_dist)
    return min_dist


def segment_poly_dist(segment, poly):
    """Distance between a segment and a rectangle."""
    # no distance if they intersect
    if segment_poly_intersect(segment, poly):
        return 0

    # otherwise we just need to consider each edge
    min_dist = np.inf
    for edge in poly.edges:
        d = segment_segment_dist(segment, edge)
        min_dist = min(d, min_dist)
    return min_dist


def swept_circle_poly_intersect(segment, radius, poly):
    """Check if a circle swept along a segment intersects a rectangle."""
    # TODO this does not seem to be working properly
    return segment_poly_dist(segment, poly) <= radius


def swept_circle_poly_intersect_time(segment, radius, poly):
    # build the padded rectangle
    circles = [Circle(center=v, radius=radius) for v in poly.vertices]
    edges = [
        Segment(start=e.start + radius * n, end=e.end + radius * n)
        for e, n in zip(poly.edges, poly.out_normals)
    ]

    # TODO this does not account for starting inside the shape!
    # but then we also need to compute the normal
    # for circle in circles:
    #     if np.linalg.norm(segment.start - circle.center) <= radius:
    #         return 0, unit(segment.start - circle.center)
    # for edge in edges:
    #     pass
    if point_poly_dist(segment.start, poly) <= radius:
        return 0, None

    # now we check each of the shapes
    # TODO this can also give us the normal of the collision
    min_time = None
    normal = None
    for circle in circles:
        if not segment_circle_intersect(segment, circle):
            continue
        t = segment_circle_intersect_time(segment, circle)
        if t < min_time:
            min_time = t
            p = segment.start + t * segment.v
            normal = unit(p - circle.center)
    for edge in edges:
        # TODO maybe the problem is that the segment actually gets slightly
        # inside the polygon
        t = segment_segment_intersect_time(segment, edge)
        if t is not None:
            if min_time is None or t < min_time:
                min_time = t
                normal = -edge.normal
    return min_time, normal
