import numpy as np
import pygame

from ..collision import Circle, point_poly_query


class Treasure(Circle):
    """Treasure to be claimed for points."""

    def __init__(self, center, radius):
        super().__init__(center, radius)
        self.color = (0, 255, 0)

    def draw(self, surface, scale=1):
        pygame.draw.circle(
            surface, self.color, scale * self.center, scale * self.radius
        )

    def update_position(self, shape, obstacles, rng):
        """Update the treasure's position to a collision-free point in
        a screen with dimensions `shape`."""
        r = self.radius * np.ones(2)
        while True:
            p = rng.uniform(low=r, high=np.array(shape) - r)
            collision = False
            for obstacle in obstacles:
                Q = point_poly_query(p, obstacle)
                if Q.distance < self.radius:
                    collision = True
                    break
            if not collision:
                break
        self.center = p
