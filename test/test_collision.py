import numpy as np

import shoot


def test_point_in_rect():
    rect = shoot.AARect(100, 100, 50, 20)

    assert shoot.point_in_poly((100, 100), rect).intersect
    assert shoot.point_in_poly((110, 110), rect).intersect

    assert not shoot.point_in_poly((110, 121), rect).intersect
    assert not shoot.point_in_poly((151, 110), rect).intersect


def test_point_segment_dist():
    segment = shoot.Segment((0, 0), (100, 0))

    assert np.isclose(shoot.point_segment_dist((50, 0), segment), 0)
    assert np.isclose(shoot.point_segment_dist((50, 10), segment), 10)
    assert np.isclose(shoot.point_segment_dist((110, 0), segment), 10)


def test_segment_segment_dist():
    # parallel
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((0, 10), (100, 10))
    assert np.isclose(shoot.segment_segment_dist(segment1, segment2), 10)

    # parallel but overlapping
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((10, 0), (110, 0))
    assert np.isclose(shoot.segment_segment_dist(segment1, segment2), 0)

    # parallel, along the same line segment, but *not* overlapping
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((110, 0), (210, 0))
    assert np.isclose(shoot.segment_segment_dist(segment1, segment2), 10)

    # intersection
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((0, 100), (100, -100))
    assert np.isclose(shoot.segment_segment_dist(segment1, segment2), 0)

    # one end point with the inside of the other segment
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((110, -50), (110, 50))
    assert np.isclose(shoot.segment_segment_dist(segment1, segment2), 10)

    # two endpoints
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((110, 10), (110, 110))
    assert np.isclose(shoot.segment_segment_dist(segment1, segment2), np.sqrt(200))


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


def test_segment_segment_intersect_time():
    # parallel
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((10, 0), (110, 0))
    assert np.isclose(shoot.segment_segment_intersect_time(segment1, segment2), 0.1)

    # parallel, reverse direction
    segment1 = shoot.Segment((0, 0), (100, 0))
    segment2 = shoot.Segment((110, 0), (10, 0))
    assert np.isclose(shoot.segment_segment_intersect_time(segment1, segment2), 0.1)

    # crossed
    segment1 = shoot.Segment((50, -50), (50, 50))
    segment2 = shoot.Segment((0, 0), (100, 0))
    assert np.isclose(shoot.segment_segment_intersect_time(segment1, segment2), 0.5)


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
