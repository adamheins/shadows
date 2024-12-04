import numpy as np

import shadows


def test_line_rect_edge_intersection():
    rect = shadows.AARect(0, 0, 100, 100)

    q = shadows.line_rect_edge_intersection(p=(50, 50), v=(1, 0), rect=rect)
    assert np.allclose(q, (100, 50))

    q = shadows.line_rect_edge_intersection(p=(50, 50), v=(-1, 0), rect=rect)
    assert np.allclose(q, (0, 50))

    q = shadows.line_rect_edge_intersection(p=(50, 50), v=(0, 1), rect=rect)
    assert np.allclose(q, (50, 100))

    q = shadows.line_rect_edge_intersection(p=(50, 50), v=(0, -1), rect=rect)
    assert np.allclose(q, (50, 0))

    q = shadows.line_rect_edge_intersection(p=(50, 50), v=(1, 1), rect=rect)
    assert np.allclose(q, (100, 100))


def test_point_in_rect():
    rect = shadows.AARect(100, 100, 50, 20)

    assert shadows.point_in_poly((100, 100), rect)
    assert shadows.point_in_poly((110, 110), rect)

    assert not shadows.point_in_poly((110, 121), rect)
    assert not shadows.point_in_poly((151, 110), rect)


def test_point_segment_query():
    segment = shadows.Segment((0, 0), (100, 0))

    assert np.isclose(shadows.point_segment_query((50, 0), segment).distance, 0)
    assert np.isclose(shadows.point_segment_query((50, 10), segment).distance, 10)
    assert np.isclose(shadows.point_segment_query((110, 0), segment).distance, 10)


def test_point_poly_query():
    rect = shadows.AARect(0, 0, 100, 100)

    Q = shadows.point_poly_query((10, 50), rect)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)
    assert np.allclose(Q.normal, [-1, 0])

    Q = shadows.point_poly_query((110, 50), rect)
    assert not Q.intersect
    assert np.isclose(Q.distance, 10)
    assert np.allclose(Q.normal, [1, 0])

    Q = shadows.point_poly_query((110, 110), rect)
    assert not Q.intersect
    assert np.isclose(Q.distance, np.sqrt(200))
    assert np.allclose(Q.normal, shadows.unit([1, 1]))


def test_segment_circle_query():
    circle = shadows.Circle(center=(0, 0), radius=10)

    # start inside, finish outside
    segment = shadows.Segment(start=(0, 0), end=(20, 0))
    Q = shadows.segment_circle_query(segment, circle)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)
    assert np.isclose(Q.time, 0)

    # start outside, finish inside
    segment = shadows.Segment(start=(-20, 0), end=(0, 0))
    Q = shadows.segment_circle_query(segment, circle)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)
    assert np.isclose(Q.time, 0.5)

    # no intersection
    segment = shadows.Segment(start=(-20, 20), end=(20, 20))
    Q = shadows.segment_circle_query(segment, circle)
    assert not Q.intersect
    assert np.isclose(Q.distance, 10)
    assert Q.time is None
    assert np.allclose(Q.normal, [0, 1])
    assert np.allclose(Q.p1, [0, 20])
    assert np.allclose(Q.p2, [0, 10])


def test_segment_segment_query():
    # parallel
    segment1 = shadows.Segment((0, 0), (100, 0))
    segment2 = shadows.Segment((0, 10), (100, 10))
    Q = shadows.segment_segment_query(segment1, segment2)
    assert not Q.intersect
    assert np.isclose(Q.distance, 10)
    assert Q.time is None

    # parallel but overlapping
    segment1 = shadows.Segment((0, 0), (100, 0))
    segment2 = shadows.Segment((10, 0), (110, 0))
    Q = shadows.segment_segment_query(segment1, segment2)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)
    assert np.isclose(Q.time, 0.1)

    # parallel, along the same line segment, but *not* overlapping
    segment1 = shadows.Segment((0, 0), (100, 0))
    segment2 = shadows.Segment((110, 0), (210, 0))
    Q = shadows.segment_segment_query(segment1, segment2)
    assert not Q.intersect
    assert np.isclose(Q.distance, 10)
    assert Q.time is None
    assert np.allclose(Q.p1, [100, 0])
    assert np.allclose(Q.p2, [110, 0])

    # intersection
    segment1 = shadows.Segment((0, 0), (100, 0))
    segment2 = shadows.Segment((0, 100), (100, -100))
    Q = shadows.segment_segment_query(segment1, segment2)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)
    assert np.isclose(Q.time, 0.5)
    assert np.allclose(Q.p1, [50, 0])
    assert np.allclose(Q.p2, [50, 0])

    # closest point is one end point with the inside of the other segment
    segment1 = shadows.Segment((0, 0), (100, 0))
    segment2 = shadows.Segment((110, -50), (110, 50))
    Q = shadows.segment_segment_query(segment1, segment2)
    assert not Q.intersect
    assert np.isclose(Q.distance, 10)
    assert Q.time is None
    assert np.allclose(Q.p1, [100, 0])
    assert np.allclose(Q.p2, [110, 0])

    # closest points are two endpoints
    segment1 = shadows.Segment((0, 0), (100, 0))
    segment2 = shadows.Segment((110, 10), (110, 110))
    Q = shadows.segment_segment_query(segment1, segment2)
    assert not Q.intersect
    assert np.isclose(Q.distance, np.sqrt(200))
    assert Q.time is None
    assert np.allclose(Q.p1, [100, 0])
    assert np.allclose(Q.p2, [110, 10])


def test_segment_poly_query():
    rect = shadows.AARect(x=0, y=0, w=100, h=100)

    # intersecting
    segment = shadows.Segment((-50, 50), (50, 50))
    Q = shadows.segment_poly_query(segment, rect)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)
    assert np.isclose(Q.time, 0.5)
    assert np.allclose(Q.normal, [-1, 0])

    # just barely intersect at one corner
    segment = shadows.Segment((50, 150), (150, 50))
    Q = shadows.segment_poly_query(segment, rect)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)
    assert np.isclose(Q.time, 0.5)

    # intersecting two edges
    segment = shadows.Segment((-50, 100), (100, -50))
    Q = shadows.segment_poly_query(segment, rect)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)
    assert np.isclose(Q.time, 1.0 / 3)
    # normal is the first edge passed through
    assert np.allclose(Q.normal, [-1, 0])

    # not intersecting
    segment = shadows.Segment((-50, 110), (50, 110))
    Q = shadows.segment_poly_query(segment, rect)
    assert not Q.intersect
    assert np.isclose(Q.distance, 10)

    segment = shadows.Segment((-20, 0), (0, -20))
    Q = shadows.segment_poly_query(segment, rect)
    assert not Q.intersect
    assert np.isclose(Q.distance, np.sqrt(200))


def test_swept_circle_poly_query():
    radius = 10
    rect = shadows.AARect(x=0, y=0, w=100, h=100)

    # not intersecting
    segment = shadows.Segment((0, 120), (100, 120))
    Q = shadows.swept_circle_poly_query(segment, radius, rect)
    assert not Q.intersect
    assert np.isclose(Q.distance, 10)

    # segment does not intersect but swept circle does
    segment = shadows.Segment((80, 130), (130, 80))
    assert not shadows.segment_poly_query(segment, rect).intersect
    Q = shadows.swept_circle_poly_query(segment, radius, rect)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)

    segment = shadows.Segment((120, 50), (80, 50))
    Q = shadows.swept_circle_poly_query(segment, radius, rect)
    assert Q.intersect
    assert np.isclose(Q.time, 0.25)

    # just touching at a single point
    segment = shadows.Segment((-50, 50), (-10, 50))
    Q = shadows.swept_circle_poly_query(segment, radius, rect)
    assert Q.intersect
    assert np.isclose(Q.time, 1)
    assert np.isclose(Q.distance, 0)
    assert np.allclose(Q.p1, [-10, 50])
    assert np.allclose(Q.p2, [-10, 50])
    assert np.allclose(Q.normal, [-1, 0])

    # some penetration
    segment = shadows.Segment((-50, 50), (0, 50))
    Q = shadows.swept_circle_poly_query(segment, radius, rect)
    assert Q.intersect
    assert np.isclose(Q.time, 0.8)
    assert np.isclose(Q.distance, 0)
    assert np.allclose(Q.normal, [-1, 0])

    # not intersecting: the closest point is when the segment passes through
    # (115, 115)
    segment = shadows.Segment((90, 140), (140, 90))
    assert not shadows.segment_poly_query(segment, rect).intersect
    Q = shadows.swept_circle_poly_query(segment, radius, rect)
    assert not Q.intersect
    assert np.isclose(Q.distance, np.sqrt(2 * 15**2) - radius)
    assert np.allclose(Q.normal, shadows.unit([1, 1]))
