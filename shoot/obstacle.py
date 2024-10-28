import numpy as np
import pygame

from .collision import AARect, orth


OBSTACLE_COLOR = (0, 0, 0)
SHADOW_COLOR = (100, 100, 100)


def _line_rect_edge_intersection(p, v, rect):
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


class Obstacle(AARect):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
        self.color = OBSTACLE_COLOR
        self.pygame_rect = pygame.Rect(x, y, w, h)

    def _compute_witness_vertices(self, point, tol=1e-3):
        right = None
        left = None
        for i in range(len(self.vertices)):
            vert = self.vertices[i]
            delta = vert - point
            normal = orth(delta)
            dists = (self.vertices - point) @ normal
            if np.all(dists >= -tol):
                right = vert
            elif np.all(dists <= tol):
                left = vert
            if left is not None and right is not None:
                break
        return right, left

    def _compute_occlusion(self, point, screen_rect, tol=1e-3):
        right, left = self._compute_witness_vertices(point, tol=tol)

        delta_right = right - point
        extra_right = _line_rect_edge_intersection(right, delta_right, screen_rect)
        normal_right = orth(delta_right)

        delta_left = left - point
        extra_left = _line_rect_edge_intersection(left, delta_left, screen_rect)
        normal_left = orth(delta_left)

        screen_dists = []
        screen_vs = []
        for v in screen_rect.vertices:
            if -(v - point) @ normal_left < 0:
                continue
            dist = (v - point) @ normal_right
            if dist >= 0:
                if len(screen_dists) > 0 and screen_dists[0] > dist:
                    screen_vs = [v, screen_vs[0]]
                    break
                else:
                    screen_dists.append(dist)
                    screen_vs.append(v)

        return [right, extra_right] + screen_vs + [extra_left, left]

    def compute_collision_normal(self, point, r):
        if point[0] >= self.x and point[0] <= self.x + self.w:
            if point[1] >= self.y - r and point[1] <= self.y + 0.5 * self.h:
                return (0, -1)
            elif point[1] <= self.y + self.h + r and point[1] > self.y + 0.5 * self.h:
                return (0, 1)
        elif point[1] >= self.y and point[1] <= self.y + self.h:
            if point[0] >= self.x - r and point[0] <= self.x + 0.5 * self.w:
                return (-1, 0)
            elif point[0] <= self.x + self.w + r and point[0] > self.x + 0.5 * self.w:
                return (1, 0)
        else:
            # closest point is one of the vertices
            v2 = np.sum((self.vertices - point) ** 2, axis=1)
            idx = np.argmin(v2)
            if v2[idx] <= r**2:
                return unit(point - self.vertices[idx, :])
        return None

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.pygame_rect)

    def draw_occlusion(self, surface, viewpoint, screen_rect):
        ps = self._compute_occlusion(viewpoint, screen_rect)
        pygame.gfxdraw.aapolygon(surface, ps, SHADOW_COLOR)
        pygame.gfxdraw.filled_polygon(surface, ps, SHADOW_COLOR)
