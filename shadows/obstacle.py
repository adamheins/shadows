import numpy as np
import pygame

from .math import orth, unit, ORTHMAT
from .collision import AARect, PaddedPoly, line_rect_edge_intersection
from .gui import Color

import time


class Obstacle(AARect):
    def __init__(self, x, y, w, h, agent_radius=None):
        super().__init__(x, y, w, h)
        self.color = Color.OBSTACLE
        self.pygame_rect = pygame.Rect(x, y, w, h)

        if agent_radius is not None:
            self.padded = PaddedPoly(self, agent_radius)

    # def __init__(self, vertices, rects):
    #     self.color = Color.OBSTACLE
    #     self.vertices = vertices
    #     pass
    #
    # def point_query(self, point):
    #     pass
    #
    # def segment_query(self, segment):
    #     pass

    def _compute_witness_vertices(self, point, tol=1e-8):
        deltas = self.vertices - point
        normals = deltas @ ORTHMAT.T
        dists = normals @ deltas.T

        right_idx = np.argmax(np.all(dists >= -tol, axis=1))
        left_idx = np.argmax(np.all(dists <= tol, axis=1))

        right = self.vertices[right_idx, :]
        left = self.vertices[left_idx, :]
        return right, left

    def _compute_occlusion2(self, point, screen_rect):
        right, left = self._compute_witness_vertices(point)

        deltas = screen_rect.vertices - point

        delta_right = unit(right - point)
        normal_right = orth(delta_right)
        dists_right = deltas @ delta_right
        extra_right = point + dists_right.max() * delta_right

        delta_left = unit(left - point)
        normal_left = -orth(delta_left)  # negative makes it inward-facing
        dists_left = deltas @ delta_left
        extra_left = point + dists_left.max() * delta_left

        # filter out the ones where either is negative, and order by increasing
        # distance from right side
        dists_right = deltas @ normal_right
        dists_left = deltas @ normal_left
        mask = (dists_right >= 0) & (dists_left >= 0)
        screen_vs_masked = screen_rect.vertices[mask]
        n = screen_vs_masked.shape[0]
        if n == 0:
            screen_vs = []
        elif n == 1:
            screen_vs = [screen_rect.vertices[mask][0]]
        else:
            dists_right_masked = dists_right[mask]

            if dists_right_masked[0] <= dists_right_masked[1]:
                screen_vs = [screen_vs_masked[0], screen_vs_masked[1]]
            else:
                screen_vs = [screen_vs_masked[1], screen_vs_masked[0]]

        return [right, extra_right] + screen_vs + [extra_left, left]

    def _compute_occlusion(self, point, screen_rect):
        right, left = self._compute_witness_vertices(point)

        deltas = screen_rect.vertices - point

        delta_right = unit(right - point)
        normal_right = orth(delta_right)
        dists_right = deltas @ delta_right
        extra_right = point + dists_right.max() * delta_right

        delta_left = unit(left - point)
        normal_left = -orth(delta_left)  # negative makes it inward-facing
        dists_left = deltas @ delta_left
        extra_left = point + dists_left.max() * delta_left

        screen_dists = []
        screen_vs = []
        for v in screen_rect.vertices:
            if (v - point) @ normal_left < 0:
                continue
            dist = (v - point) @ normal_right
            if dist >= 0:
                # we know that at most two screen vertices can be included
                if len(screen_dists) > 0 and screen_dists[0] > dist:
                    screen_vs = [v, screen_vs[0]]
                    break
                else:
                    screen_dists.append(dist)
                    screen_vs.append(v)

        return [right, extra_right] + screen_vs + [extra_left, left]

    def draw(self, surface, scale=1):
        rect = pygame.Rect(
            scale * self.x, scale * self.y, scale * self.w, scale * self.h
        )
        pygame.draw.rect(surface, self.color, rect)

    def draw_occlusion(self, surface, viewpoint, screen_rect, scale=1):
        ps = self._compute_occlusion2(viewpoint, screen_rect)
        pygame.draw.polygon(surface, Color.SHADOW, [scale * p for p in ps])
        # pygame.gfxdraw.aapolygon(surface, ps, Color.SHADOW)
        # pygame.gfxdraw.filled_polygon(surface, ps, Color.SHADOW)
