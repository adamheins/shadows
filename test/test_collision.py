import numpy as np

import shoot


def test_point_in_rect():
    rect = shoot.AARect(100, 100, 50, 20)

    assert shoot.point_in_rect((100, 100), rect)
    assert shoot.point_in_rect((110, 110), rect)

    assert not shoot.point_in_rect((110, 121), rect)
    assert not shoot.point_in_rect((151, 110), rect)


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


def test_segment_rect_intersect():
    rect = shoot.AARect(100, 100, 100, 100)

    # not intersecting
    segment = shoot.Segment((100, 90), (150, 90))
    assert not shoot.segment_rect_intersect(segment, rect)

    segment = shoot.Segment((190, 210), (220, 190))
    assert not shoot.segment_rect_intersect(segment, rect)

    # intersecting
    segment = shoot.Segment((90, 90), (110, 110))
    assert shoot.segment_rect_intersect(segment, rect)

    segment = shoot.Segment((180, 210), (210, 180))
    assert shoot.segment_rect_intersect(segment, rect)
