import numpy as np

from .math import unit, orth, quad_formula


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


class Polygon:
    """2D polygon."""

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
    """2D line segment."""

    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)

        self.v = self.end - self.start
        self.direction = unit(self.v)
        self.normal = orth(self.direction)

    def __repr__(self):
        return f"Segment(start={self.start}, end={self.end})"


class Circle:
    """2D circle."""

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
        Rectangle containing ``p``.

    Returns
    -------
    : pair of float
        The closest intersection point corresponding to non-negative distance
        along direction v.
    """
    ts = []

    # vertical edges
    if not np.isclose(v[0], 0):
        ts.extend([(rect.x - p[0]) / v[0], (rect.x + rect.w - p[0]) / v[0]])

    # horizontal edges
    if not np.isclose(v[1], 0):
        ts.extend([(rect.y - p[1]) / v[1], (rect.y + rect.h - p[1]) / v[1]])

    # return the smallest positive value
    ts = np.array(ts)
    t = np.min(ts[ts >= 0])
    return p + t * np.array(v)


def point_in_poly(point, poly, tol=1e-8):
    """Check if a point is inside a polygon.

    Parameters
    ----------
    point : pair of float
        A 2D point.
    poly : Polygon
        A polygon.
    tol : float
        Tolerance for the point to be considered inside the polygon. The point
        is still considered inside the polygon as long as its distance to the
        shape is less than ``tol``.

    Returns
    -------
    : bool
        True if the polygon contains the point, False otherwise.
    """
    for v, n in zip(poly.vertices, poly.in_normals):
        if (point - v) @ n < -tol:
            return False
    return True


def point_in_rect(point, rect, tol=1e-8):
    """Check if a point is inside a rectangle.

    Parameters
    ----------
    point : pair of float
        A 2D point.
    rect : AARect
        A rectangle.
    tol : float
        Tolerance for the point to be considered inside the polygon.

    Returns
    -------
    : bool
        True if the rectangle contains the point, False otherwise.
    """
    x, y = point
    return (
        x >= rect.x - tol
        and x <= rect.x + rect.w + tol
        and y >= rect.y - tol
        and y <= rect.y + rect.h + tol
    )


def point_circle_query(point, circle):
    """Collision query between a point and a circle.

    Parameters
    ----------
    point : pair of float
        A 2D point.
    circle : Circle
        A circle.

    Returns
    -------
    : CollisionQuery
        The collision information between the two shapes.
    """
    d = np.linalg.norm(point - circle.center) - circle.radius
    n = unit(point - circle.center)

    # the point is inside the circle
    if d <= 0:
        return CollisionQuery(distance=0, p1=point, p2=point, normal=n, intersect=True)

    p2 = circle.center + circle.radius * n
    return CollisionQuery(distance=d, p1=point, p2=p2, normal=n, intersect=False)


def point_segment_query(point, segment):
    """Collision query between a point and a line segment.

    Parameters
    ----------
    point : pair of float
        A 2D point.
    segment : Segment
        A line segment.

    Returns
    -------
    : CollisionQuery
        The collision information between the two shapes.
    """
    q = np.array(segment.start - point)

    t = -(q @ segment.v) / (segment.v @ segment.v)
    if t >= 0 and t <= 1:
        r = segment.start + t * segment.v
        d = np.linalg.norm(point - r)
        intersect = np.isclose(d, 0)
        return CollisionQuery(distance=d, p1=point, p2=r, intersect=intersect)

    d1 = np.linalg.norm(point - segment.start)
    d2 = np.linalg.norm(point - segment.end)
    if d1 < d2:
        n = unit(point - segment.start)
        return CollisionQuery(distance=d1, normal=n, p1=point, p2=segment.start)
    n = unit(point - segment.end)
    return CollisionQuery(distance=d2, normal=n, p1=point, p2=segment.end)


def point_poly_query(point, poly):
    """Collision query between a point and a polygon.

    Parameters
    ----------
    point : pair of float
        A 2D point.
    poly : Polygon
        A polygon.

    Returns
    -------
    : CollisionQuery
        The collision information between the two shapes.
    """
    # inward-facing depth values for each edge
    depths = np.array([(point - v) @ n for v, n in zip(poly.vertices, poly.in_normals)])
    min_idx = np.argmin(depths)

    # if all depths are positive, then the point is inside the polygon
    if depths[min_idx] >= 0:
        normal = poly.out_normals[min_idx]
        return CollisionQuery(
            distance=0, p1=point, p2=point, normal=normal, intersect=True
        )

    # detect if a vertex is the closest point
    # we need only check the vertices on the closest edge
    n = len(poly.vertices)
    prev_idx = (min_idx - 1) % n
    next_idx = (min_idx + 1) % n
    if depths[prev_idx] < 0 or depths[next_idx] < 0:
        if depths[prev_idx] < 0:
            v = poly.vertices[min_idx]
        else:
            v = poly.vertices[next_idx]
        dist = np.linalg.norm(point - v)
        normal = unit(point - v)
        return CollisionQuery(
            distance=dist, p1=point, p2=v, normal=normal, intersect=False
        )

    # otherwise we know the closest point lies on the segment
    Q = point_segment_query(point, poly.edges[min_idx])
    Q.normal = poly.out_normals[min_idx]
    return Q


def segment_circle_query(segment, circle):
    """Collision query between a segment and a circle.

    Parameters
    ----------
    segment : Segment
        A line segment.
    circle : Circle
        A circle.

    Returns
    -------
    : CollisionQuery
        The collision information between the two shapes.
    """
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
        # compute the intersection time
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


def _segments_are_parallel(segment1, segment2, tol=1e-8):
    """True if the segments are parallel, False otherwise."""
    return np.isclose(segment1.normal @ segment2.v, 0, atol=tol, rtol=0)


def segment_segment_query(segment1, segment2):
    """Collision query between two line segments.

    Parameters
    ----------
    segment1 : Segment
        The first segment.
    segment2 : Segment
        The second segment.

    Returns
    -------
    : CollisionQuery
        The collision information between the two shapes. The normal is only
        provided when the shapes are not intersecting. The time is provided
        with respect to the first segment, if they are interecting.
    """
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


def _segment_poly_intersect(segment, poly):
    """True if segment and rectangle are intersecting, False otherwise."""
    # look for separating axis
    normals = np.vstack((poly.out_normals, segment.normal))
    for normal in normals:
        s = [normal @ segment.start, normal @ segment.end]
        r = poly.vertices @ normal
        if np.max(s) < np.min(r) or np.min(s) > np.max(r):
            return False
    return True


def segment_poly_query(segment, poly):
    """Collision query between a segment and a polygon.

    Parameters
    ----------
    segment : Segment
        A line segment.
    poly : Polygon
        A polygon.

    Returns
    -------
    : CollisionQuery
        The collision information between the two shapes.
    """
    if _segment_poly_intersect(segment, poly):
        # if the segment starts in the polygon, then we're done
        Q = point_poly_query(segment.start, poly)
        if Q.intersect:
            Q.time = 0
            return Q

        # otherwise, the segment must intersect an edge
        min_time_query = segment_segment_query(segment, poly.edges[0])
        min_time_query.normal = poly.out_normals[0]
        for i, edge in enumerate(poly.edges[1:]):
            Q = segment_segment_query(segment, edge)
            if Q.intersect and (
                min_time_query.time is None or Q.time < min_time_query.time
            ):
                Q.normal = poly.out_normals[i + 1]
                min_time_query = Q
        return min_time_query

    # not intersecting: we look for the closest edge
    min_dist_query = segment_segment_query(segment, poly.edges[0])
    min_dist_query.normal = poly.out_normals[0]
    for i, edge in enumerate(poly.edges[1:]):
        Q = segment_segment_query(segment, edge)
        if Q.distance < min_dist_query.distance:
            Q.normal = poly.out_normals[i + 1]
            min_dist_query = Q
    return min_dist_query


def swept_circle_poly_query(segment, radius, poly):
    """Collision query between circle swept along a path and a polygon.

    See https://gamedev.stackexchange.com/a/106032.

    Parameters
    ----------
    segment : Segment
        A line segment representing the path of the circle's center.
    radius : float
        The radius of the circle.
    poly : Polygon
        A polygon.

    Returns
    -------
    : CollisionQuery
        The collision information between the two shapes.
    """

    # check if the shapes do not intersect at all
    Q = segment_poly_query(segment, poly)
    if Q.distance > radius:
        n = unit(Q.p2 - Q.p1)
        p1 = Q.p1 + radius * n
        d = Q.distance - radius
        return CollisionQuery(distance=d, normal=-n, p1=p1, p2=Q.p2, intersect=False)

    # build the padded rectangle
    circles = [Circle(center=v, radius=radius) for v in poly.vertices]
    edges = [
        Segment(start=e.start + radius * n, end=e.end + radius * n)
        for e, n in zip(poly.edges, poly.out_normals)
    ]

    # check if we are starting inside the shape
    vertices = []
    for edge in edges:
        vertices.append(edge.start)
        vertices.append(edge.end)
    poly2 = Polygon(vertices)

    Q = point_poly_query(segment.start, poly2)
    if Q.intersect:
        Q.time = 0
        return Q

    for circle in circles:
        Q = point_circle_query(segment.start, circle)
        if Q.intersect:
            Q.time = 0
            return Q

    # find the minimum intersection time with all of the shapes
    min_time_query = segment_circle_query(segment, circles[0])
    for circle in circles:
        Q = segment_circle_query(segment, circle)
        if Q.intersect and (
            min_time_query.time is None or Q.time < min_time_query.time
        ):
            min_time_query = Q

    for i, edge in enumerate(edges):
        Q = segment_segment_query(segment, edge)
        if Q.intersect and (
            min_time_query.time is None or Q.time < min_time_query.time
        ):
            Q.normal = poly.out_normals[i]
            min_time_query = Q

    return min_time_query
