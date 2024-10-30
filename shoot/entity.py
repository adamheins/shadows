import numpy as np
import pygame

from .collision import Circle, Segment

# TODO how to handle these constants?

PLAYER_VELOCITY = 200  # px per second
BULLET_VELOCITY = 500

SHOT_COOLDOWN_TICKS = 20
RELOAD_TICKS = 120

PROJECTILE_COLOR = (0, 0, 0)
PROJECTILE_RADIUS = 3

PLAYER_COLOR = (255, 0, 0)
ENEMY_COLOR = (0, 0, 255)
AGENT_RADIUS = 10

CLIP_SIZE = 1
MAX_HEALTH = 1


class Entity:
    """Dynamic object in the game."""

    _current_id = 0

    def __init__(self, position, velocity=None):
        self.position = np.array(position)

        if velocity is None:
            velocity = np.zeros_like(position)
        self.velocity = velocity

        self.id = Entity._current_id
        Entity._current_id += 1


class Agent(Entity):
    def __init__(self, position, color):
        super().__init__(position)

        self.color = color
        self.radius = AGENT_RADIUS

        self.health = MAX_HEALTH
        self.ammo = CLIP_SIZE
        self.shot_cooldown = 0
        self.reload_ticks = 0

    @classmethod
    def player(cls, position):
        return cls(position, color=PLAYER_COLOR)

    @classmethod
    def enemy(cls, position):
        return cls(position, color=ENEMY_COLOR)

    def draw(self, surface):
        pygame.gfxdraw.aacircle(
            surface,
            int(self.position[0]),
            int(self.position[1]),
            int(self.radius),
            self.color,
        )
        pygame.gfxdraw.filled_circle(
            surface,
            int(self.position[0]),
            int(self.position[1]),
            int(self.radius),
            self.color,
        )

    def move(self, velocity):
        self.velocity = self.velocity + velocity

    def step(self, dt):
        self.shot_cooldown = max(0, self.shot_cooldown - 1)

        # if done reloading, fill clip
        if self.reload_ticks == 1:
            self.ammo = CLIP_SIZE
        self.reload_ticks = max(0, self.reload_ticks - 1)

        self.position = self.position + dt * self.velocity
        self.velocity = np.zeros(2)

    def reload(self):
        """Reload ammo magazine."""
        self.reload_ticks = RELOAD_TICKS

    def shoot(self, target):
        """Shoot a projectile at the given target."""
        # cannot shoot if agent has shot too recently or is reloading
        if self.shot_cooldown > 0 or self.reload_ticks > 0:
            return None

        # if the target is inside the agent, do nothing
        norm = np.linalg.norm(target - self.position)
        if norm > self.radius:
            direction = (target - self.position) / norm
            self.shot_cooldown = SHOT_COOLDOWN_TICKS
            self.ammo -= 1
            if self.ammo == 0:
                self.reload()
            self.velocity -= 0.5 * PLAYER_VELOCITY * direction
            return Projectile(
                position=self.position.copy(),
                velocity=BULLET_VELOCITY * direction,
                agent_id=self.id,
            )
        else:
            return None

    def circle(self):
        """Generate a bounding circle at the agent's current position."""
        return Circle(self.position, self.radius)

    def compute_view(self):
        pass

    def draw_view_occlusion(self, surface):
        pass


class Projectile(Entity):
    """A bullet fired by an agent."""

    def __init__(self, position, velocity, agent_id):
        super().__init__(position, velocity)
        self.agent_id = agent_id
        self.color = PROJECTILE_COLOR
        self.radius = PROJECTILE_RADIUS

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.position, self.radius)

    def path(self, dt):
        return Segment(self.position, self.position + dt * self.velocity)

    def step(self, dt):
        self.position += dt * self.velocity


class Action:
    """Action for one agent."""

    def __init__(self, velocity, target, reload=False):
        self.velocity = velocity
        self.target = target
        self.reload = reload
