import numpy as np

from .math import unit, orth, quad_formula


# TODO bring this back again?
class CollisionQuery:
    def __init__(
        self, distance=None, time=None, normal=None, p1=None, p2=None, intersect=False
    ):
        self.p1 = p1
        self.p2 = p2
        self.normal = normal
        self.distance = distance
        self.time = time
        self.intersect = intersect

    def __repr__(self):
        return f"CollisionQuery(distance={self.distance}, time={self.time}, normal={self.normal}, p1={self.p1}, p2={self.p2}, intersect={self.intersect})"


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
            return False
    return True


def point_in_rect(point, rect):
    """Specialized version for rectangles."""
    x, y = point
    return x >= rect.x and x <= rect.x + rect.w and y >= rect.y and y <= rect.y + rect.h


def point_circle_query(point, circle):
    """Collision query between a point and a circle."""
    d = np.linalg.norm(segment.start - circle.center) - circle.radius
    n = unit(point - circle.center)

    # the point is inside the circle
    if d <= 0:
        return CollisionQuery(distance=0, p1=point, p2=point, normal=n, intersect=True)

    p2 = circle.center + circle.radius * n
    return CollisionQuery(distance=d, p1=point, p2=p2, normal=n, intersect=False)


def point_segment_query(point, segment):
    """Collision query between a point and a line segment."""
    q = np.array(segment.start - point)

    t = -(q @ segment.v) / (segment.v @ segment.v)
    if t >= 0 and t <= 1:
        r = segment.start + t * segment.v
        d = np.linalg.norm(point - r)
        intersect = np.isclose(d, 0)
        return CollisionQuery(distance=d, p1=point, p2=r, intersect=intersect)
        # return np.linalg.norm(point - r)
    d1 = np.linalg.norm(point - segment.start)
    d2 = np.linalg.norm(point - segment.end)
    if d1 < d2:
        return CollisionQuery(distance=d1, p1=point, p2=segment.start)
    return CollisionQuery(distance=d2, p1=point, p2=segment.end)


def point_poly_query(point, poly):
    """Collision query between a point and a polygon."""
    min_depth = np.inf
    normal = None
    for v, n in zip(poly.vertices, poly.in_normals):
        depth = (point - v) @ n
        if depth < min_depth:
            min_depth = depth
            normal = -n

    # depth must be non-negative for all normals for point to be inside the
    # polygon
    if min_depth >= 0:
        return CollisionQuery(
            distance=0, p1=point, p2=point, normal=normal, intersect=True
        )

    # otherwise the point is outside the polygon
    # return the query for the closest edge
    min_dist_query = point_segment_query(point, poly.edges[0])
    for edge in poly.edges[1:]:
        Q = point_segment_query(point, edge)
        if Q.distance < min_dist_query.distance:
            Q.normal = unit(point - Q.p2)
            min_dist_query = Q
    return min_dist_query


def segment_circle_query(segment, circle):
    Qc = point_segment_query(circle.center, segment)
    normal = unit(Qc.p2 - circle.center)

    # segment and circle do not intersect
    if Qc.distance >= circle.radius:
        p2 = circle.center + circle.radius * normal
        d = Qc.distance - circle.radius
        return CollisionQuery(
            distance=d, p1=Qc.p2, p2=p2, normal=normal, intersect=False
        )

    # segment and circle intersect
    if np.linalg.norm(segment.start - circle.center) <= circle.radius:
        # if the segment starts inside the circle, the time is 0
        t = 0
    else:
        q = segment.start - circle.center
        v = segment.v

        a = v @ v
        b = 2 * q @ v
        c = q @ q - circle.radius**2
        ts = quad_formula(a, b, c)
        t = np.min(ts)

    return CollisionQuery(
        distance=0, p1=Qc.p2, p2=Qc.p2, normal=normal, time=t, intersect=True
    )


# def segment_circle_intersect(segment, circle):
#     """True if circle and line segment intersect, False otherwise."""
#     return point_segment_dist(circle.center, segment).distance <= circle.radius
#
#
# def segment_circle_intersect_time(segment, circle):
#     # the segment starts in the circle already
#     if np.linalg.norm(segment.start - circle.center) <= circle.radius:
#         return 0
#
#     q = segment.start - circle.center
#     v = segment.v
#
#     a = v @ v
#     b = 2 * q @ v
#     c = q @ q - circle.radius**2
#     ts = quad_formula(a, b, c)
#     return np.min(ts)


def _segments_are_parallel(segment1, segment2):
    return np.isclose(segment1.normal @ segment2.v, 0)


def segment_segment_query(segment1, segment2):
    if not _segments_are_parallel(segment1, segment2):
        v1 = segment1.v
        v2 = segment2.v
        d = segment1.start - segment2.start

        # determine the intersection point for the infinite lines
        A = np.array([[-v1 @ v1, v1 @ v2], [-v1 @ v2, v2 @ v2]])
        b = np.array([v1 @ d, v2 @ d])
        t = np.linalg.solve(A, b)

        # line segments actually intersect, so we are down
        if 0 <= t[0] <= 1 and 0 <= t[1] <= 1:
            p = segment1.start + t[0] * segment1.v
            return CollisionQuery(distance=0, p1=p, p2=p, time=t[0], intersect=True)
    elif np.isclose(segment1.normal @ (segment2.start - segment1.start), 0):
        # parallel and along the same line
        direction = segment1.direction
        p0 = segment1.start
        t1 = ((segment2.start - p0) @ direction) / (segment1.v @ direction)
        t2 = ((segment2.end - p0) @ direction) / (segment1.v @ direction)

        t = None
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            t = min(t1, t2)
            p = segment2.start
        elif 0 <= t1 <= 1:
            t = t1
            p = segment2.start
        elif 0 <= t2 <= 1:
            t = t2
            p = segment2.end

        if t is not None:
            return CollisionQuery(distance=0, p1=p, p2=p, time=t, intersect=True)

    # if the lines are not parallel and/or do not intersect, then at least one
    # of the closest points must be an endpoint: check all four
    min_dist_query = point_segment_query(segment1.start, segment2)

    Q = point_segment_query(segment1.end, segment2)
    if Q.distance < min_dist_query.distance:
        min_dist_query = Q

    Q = point_segment_query(segment2.start, segment1)
    if Q.distance < min_dist_query.distance:
        Q.p1, Q.p2 = Q.p2, Q.p1
        min_dist_query = Q

    Q = point_segment_query(segment2.end, segment1)
    if Q.distance < min_dist_query.distance:
        Q.p1, Q.p2 = Q.p2, Q.p1
        min_dist_query = Q

    return min_dist_query


def segment_poly_query(segment, poly):
    pass


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
    if point_in_poly(segment.start, poly):
        return 0

    min_time = None
    for edge in poly.edges:
        t = segment_segment_query(segment, edge).time
        if t is not None:
            if min_time is None:
                min_time = t
            else:
                min_time = min(t, min_time)
    return min_time


def segment_poly_dist(segment, poly):
    """Distance between a segment and a rectangle."""
    # no distance if they intersect
    if segment_poly_intersect(segment, poly):
        return 0

    # otherwise we just need to consider each edge
    min_dist = np.inf
    for edge in poly.edges:
        d = segment_segment_query(segment, edge).distance
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
