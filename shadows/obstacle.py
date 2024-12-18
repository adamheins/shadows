import numpy as np
import pygame

from .math import orth, unit, ORTHMAT
from .collision import AARect, line_rect_edge_intersection
from .gui import Color

import time


class Obstacle(AARect):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
        self.color = Color.OBSTACLE
        self.pygame_rect = pygame.Rect(x, y, w, h)

    # def _compute_witness_vertices(self, point, tol=1e-8):
    #     right = None
    #     left = None
    #     for i in range(len(self.vertices)):
    #         vert = self.vertices[i]
    #         delta = vert - point
    #         normal = orth(delta)
    #         dists = (self.vertices - point) @ normal
    #         if np.all(dists >= -tol):
    #             right = vert
    #         elif np.all(dists <= tol):
    #             left = vert
    #         if left is not None and right is not None:
    #             break
    #     return right, left

    def _compute_witness_vertices(self, point, tol=1e-8):
        deltas = self.vertices - point
        normals = deltas @ ORTHMAT.T
        dists = normals @ deltas.T

        right_idx = np.argmax(np.all(dists >= -tol, axis=1))
        left_idx = np.argmax(np.all(dists <= tol, axis=1))

        right = self.vertices[right_idx, :]
        left = self.vertices[left_idx, :]
        return right, left

    def _compute_occlusion(self, point, screen_rect):
        right, left = self._compute_witness_vertices(point)

        delta_right = unit(right - point)
        normal_right = orth(delta_right)
        dists_right = (screen_rect.vertices - point) @ delta_right
        extra_right = point + max(dists_right) * delta_right

        delta_left = unit(left - point)
        normal_left = -orth(delta_left)  # negative makes it inward-facing
        dists_left = (screen_rect.vertices - point) @ delta_left
        extra_left = point + max(dists_left) * delta_left

        # TODO need to filter out the ones where either is negative, and order
        # by increasing distance from right? side
        # dists_right = (screen_rect.vertices - point) @ normal_right
        # dists_left = (screen_rect.vertices - point) @ normal_left

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
        ps = self._compute_occlusion(viewpoint, screen_rect)
        pygame.draw.polygon(surface, Color.SHADOW, [scale * p for p in ps])
        # pygame.gfxdraw.aapolygon(surface, ps, Color.SHADOW)
        # pygame.gfxdraw.filled_polygon(surface, ps, Color.SHADOW)
