import numpy as np
import pygame

from .math import rotmat, orth, unit, wrap_to_pi, angle2pi
from .collision import Circle, Segment, line_rect_edge_intersection
from .gui import Color


PLAYER_FORWARD_VEL = 150  # px per second
PLAYER_BACKWARD_VEL = 50  # px per second
PLAYER_IT_VEL = 100  # px per second
PLAYER_ANGVEL = 5  # rad per second
VIEW_ANGLE = np.pi / 3

BULLET_VELOCITY = 500

SHOT_COOLDOWN_TICKS = 20
RELOAD_TICKS = 120

PROJECTILE_RADIUS = 3
AGENT_RADIUS = 5

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
    def __init__(self, position, color, angle=0, it=False):
        super().__init__(position)

        self.color = color
        self.radius = AGENT_RADIUS

        # in radians, relative to positive x-axis
        self.angle = angle
        self.angvel = 0

        self.health = MAX_HEALTH
        self.ammo = CLIP_SIZE
        self.shot_cooldown = 0
        self.reload_ticks = 0

        # "it" in a game of tag
        self.it = it
        self.viewtarget = None

    @classmethod
    def player(cls, position, it=False):
        return cls(position, color=Color.PLAYER, it=it)

    @classmethod
    def enemy(cls, position, it=False):
        return cls(position, color=Color.ENEMY, it=it)

    def rotmat(self):
        return rotmat(self.angle)

    def direction(self):
        return self.rotmat()[:, 0]

    def draw(self, surface):
        if self.it:
            pygame.draw.circle(surface, (255, 255, 255), self.position, self.radius + 2)
        pygame.draw.circle(surface, self.color, self.position, self.radius)
        endpoint = self.position + self.radius * rotmat(self.angle)[:, 0]
        pygame.draw.line(surface, (0, 0, 0), self.position, endpoint, 1)

    def command(self, action):
        if action.viewtarget is not None:
            self.viewtarget = action.viewtarget

        projectile = None
        if action.target is not None:
            projectile = agent.shoot(action.target)

        if action.reload:
            agent.reload()

        # different speed when "it"
        if self.it:
            forward_vel = PLAYER_IT_VEL
        else:
            forward_vel = PLAYER_FORWARD_VEL

        self.angvel = PLAYER_ANGVEL * unit(action.angdir)
        if action.frame == Action.LOCAL:
            if action.lindir[0] >= 0:
                linvel = forward_vel * unit(action.lindir)
            else:
                linvel = PLAYER_BACKWARD_VEL * unit(action.lindir)
            vel = rotmat(self.angle) @ linvel
        else:
            vel = forward_vel * unit(action.lindir)
        self.velocity = vel

        return projectile

    def step(self, dt):
        self.shot_cooldown = max(0, self.shot_cooldown - 1)

        # if done reloading, fill clip
        if self.reload_ticks == 1:
            self.ammo = CLIP_SIZE
        self.reload_ticks = max(0, self.reload_ticks - 1)

        self.angle = wrap_to_pi(self.angle + dt * self.angvel)
        self.position = self.position + dt * self.velocity

        self.velocity = np.zeros(2)
        self.angvel = 0

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
            self.velocity -= PLAYER_BACKWARD_VEL * direction
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

    def _compute_view_occlusion(self, screen_rect, viewtarget):
        if viewtarget is None:
            angle = self.angle
        else:
            # TODO compute angle to viewtarget
            r = viewtarget - self.position
            angle = np.arctan2(-r[1], r[0])

        vr = rotmat(angle + VIEW_ANGLE) @ [1, 0]
        vl = rotmat(angle - VIEW_ANGLE) @ [1, 0]

        extra_right = line_rect_edge_intersection(self.position, vr, screen_rect)
        extra_left = line_rect_edge_intersection(self.position, vl, screen_rect)

        screen_dists = []
        screen_vs = []
        for v in screen_rect.vertices:
            r = v - self.position

            # compute angle and don't wrap it around pi
            a = angle2pi(r, start=angle)

            # only consider the vertices *not* in the player's view
            if a >= VIEW_ANGLE and a <= 2 * np.pi - VIEW_ANGLE:
                screen_dists.append(a)
                screen_vs.append(v)

        # sort vertices in order of increasing angle
        idx = np.argsort(screen_dists)
        screen_vs = [screen_vs[i] for i in idx]
        return [self.position, extra_right] + screen_vs + [extra_left]

    def draw_view_occlusion(self, surface, screen_rect):
        if self.it:
            return
        ps = self._compute_view_occlusion(screen_rect, viewtarget=None)
        pygame.draw.polygon(surface, Color.SHADOW, ps)
        # pygame.gfxdraw.aapolygon(surface, ps, Color.SHADOW)
        # pygame.gfxdraw.filled_polygon(surface, ps, Color.SHADOW)


class Projectile(Entity):
    """A bullet fired by an agent."""

    def __init__(self, position, velocity, agent_id):
        super().__init__(position, velocity)
        self.agent_id = agent_id
        self.color = Color.PROJECTILE
        self.radius = PROJECTILE_RADIUS

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.position, self.radius)

    def path(self, dt):
        return Segment(self.position, self.position + dt * self.velocity)

    def step(self, dt):
        self.position += dt * self.velocity


class Action:
    """Action for one agent."""

    WORLD = 0
    LOCAL = 1

    def __init__(
        self, lindir, angdir=0, target=None, reload=False, frame=WORLD, viewtarget=None
    ):
        self.frame = frame
        self.lindir = lindir
        self.angdir = angdir
        self.target = target
        self.reload = reload
        self.viewtarget = viewtarget
