import numpy as np

import shoot


def test_point_in_rect():
    rect = shoot.AARect(100, 100, 50, 20)

    assert shoot.point_in_poly((100, 100), rect)
    assert shoot.point_in_poly((110, 110), rect)

    assert not shoot.point_in_poly((110, 121), rect)
    assert not shoot.point_in_poly((151, 110), rect)


def test_point_segment_query():
    segment = shoot.Segment((0, 0), (100, 0))

    assert np.isclose(shoot.point_segment_query((50, 0), segment).distance, 0)
    assert np.isclose(shoot.point_segment_query((50, 10), segment).distance, 10)
    assert np.isclose(shoot.point_segment_query((110, 0), segment).distance, 10)


def test_point_poly_query():
    rect = shoot.AARect(0, 0, 100, 100)

    Q = shoot.point_poly_query((10, 50), rect)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)
    assert np.allclose(Q.normal, [-1, 0])

    Q = shoot.point_poly_query((110, 50), rect)
    assert not Q.intersect
    assert np.isclose(Q.distance, 10)
    assert np.allclose(Q.normal, [1, 0])

    Q = shoot.point_poly_query((110, 110), rect)
    assert not Q.intersect
    assert np.isclose(Q.distance, np.sqrt(200))
    assert np.allclose(Q.normal, shoot.unit([1, 1]))


def test_segment_circle_query():
    circle = shoot.Circle(center=(0, 0), radius=10)

    # start inside, finish outside
    segment = shoot.Segment(start=(0, 0), end=(20, 0))
    Q = shoot.segment_circle_query(segment, circle)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)
    assert np.isclose(Q.time, 0)

    # start outside, finish inside
    segment = shoot.Segment(start=(-20, 0), end=(0, 0))
    Q = shoot.segment_circle_query(segment, circle)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)
    assert np.isclose(Q.time, 0.5)

    # no intersection
    segment = shoot.Segment(start=(-20, 20), end=(20, 20))
    Q = shoot.segment_circle_query(segment, circle)
    assert not Q.intersect
    assert np.isclose(Q.distance, 10)
    assert Q.time is None
    assert np.allclose(Q.normal, [0, 1])
    assert np.allclose(Q.p1, [0, 20])
    assert np.allclose(Q.p2, [0, 10])


def test_segment_segment_query():
    # parallel
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((0, 10), (100, 10))
    Q = shoot.segment_segment_query(segment1, segment2)
    assert not Q.intersect
    assert np.isclose(Q.distance, 10)
    assert Q.time is None

    # parallel but overlapping
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((10, 0), (110, 0))
    Q = shoot.segment_segment_query(segment1, segment2)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)
    assert np.isclose(Q.time, 0.1)

    # parallel, along the same line segment, but *not* overlapping
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((110, 0), (210, 0))
    Q = shoot.segment_segment_query(segment1, segment2)
    assert not Q.intersect
    assert np.isclose(Q.distance, 10)
    assert Q.time is None
    assert np.allclose(Q.p1, [100, 0])
    assert np.allclose(Q.p2, [110, 0])

    # intersection
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((0, 100), (100, -100))
    Q = shoot.segment_segment_query(segment1, segment2)
    assert Q.intersect
    assert np.isclose(Q.distance, 0)
    assert np.isclose(Q.time, 0.5)
    assert np.allclose(Q.p1, [50, 0])
    assert np.allclose(Q.p2, [50, 0])

    # closest point is one end point with the inside of the other segment
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((110, -50), (110, 50))
    Q = shoot.segment_segment_query(segment1, segment2)
    assert not Q.intersect
    assert np.isclose(Q.distance, 10)
    assert Q.time is None
    assert np.allclose(Q.p1, [100, 0])
    assert np.allclose(Q.p2, [110, 0])

    # closest points are two endpoints
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((110, 10), (110, 110))
    Q = shoot.segment_segment_query(segment1, segment2)
    assert not Q.intersect
    assert np.isclose(Q.distance, np.sqrt(200))
    assert Q.time is None
    assert np.allclose(Q.p1, [100, 0])
    assert np.allclose(Q.p2, [110, 10])


def test_segment_poly_intersect():
    rect = shoot.AARect(100, 100, 100, 100)

    # not intersecting
    segment = shoot.Segment((100, 90), (150, 90))
    assert not shoot.segment_poly_intersect(segment, rect)

    segment = shoot.Segment((190, 210), (220, 190))
    assert not shoot.segment_poly_intersect(segment, rect)

    # intersecting
    segment = shoot.Segment((90, 90), (110, 110))
    assert shoot.segment_poly_intersect(segment, rect)

    segment = shoot.Segment((180, 210), (210, 180))
    assert shoot.segment_poly_intersect(segment, rect)


def test_segment_poly_intersect_time():
    rect = shoot.AARect(x=0, y=0, w=100, h=100)

    segment = shoot.Segment((-50, 50), (50, 50))
    assert np.isclose(shoot.segment_poly_intersect_time(segment, rect), 0.5)

    # just barely intersect at one corner
    segment = shoot.Segment((50, 150), (150, 50))
    assert np.isclose(shoot.segment_poly_intersect_time(segment, rect), 0.5)

    # not intersecting
    segment = shoot.Segment((-50, 110), (50, 110))
    assert shoot.segment_poly_intersect_time(segment, rect) is None


def swept_circle_poly_intersect():
    radius = 10
    rect = shoot.AARect(x=0, y=0, w=100, h=100)

    # not intersecting
    segment = shoot.Segment((0, 120), (100, 120))
    assert not shoot.swept_circle_poly_intersect(segment, radius, rect)

    # segment does not intersect but swept circle does
    segment = shoot.Segment((80, 130), (130, 80))
    assert not shoot.segment_poly_intersect(segment, radius, rect)
    assert shoot.swept_circle_poly_intersect(segment, radius, rect)

    segment = shoot.Segment((105, 105), (105, 150))
    assert not shoot.segment_poly_intersect(segment, radius, rect)
    assert shoot.swept_circle_poly_intersect(segment, radius, rect)
