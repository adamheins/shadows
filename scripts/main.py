#!/usr/bin/env python3
import pygame
import pygame.gfxdraw
import numpy as np
import pymunk
import pymunk.pygame_util

from shoot import *


SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
SCREEN_VERTS = np.array(
    [[0, 0], [SCREEN_WIDTH, 0], [SCREEN_WIDTH, SCREEN_HEIGHT], [0, SCREEN_HEIGHT]]
)

BACKGROUND_COLOR = (219, 200, 184)
OBSTACLE_COLOR = (0, 0, 0)

FRAMERATE = 60
PHYSICS_STEP_PER_TICK = 10
TIMESTEP = 1.0 / (FRAMERATE * PHYSICS_STEP_PER_TICK)

PLAYER_VELOCITY = 200  # px per second
BULLET_VELOCITY = 1000

PLAYER_MASS = 10
BULLET_MASS = 0.1

CLIP_SIZE = 20  # bullets per reload
SHOT_COOLDOWN_TICKS = 1
RELOAD_TICKS = 2 * FRAMERATE


class Text:
    """Text label."""

    def __init__(self, text, font, position, color):
        self.text = text
        self.font = font
        self.color = color
        self.position = position

        self.update()

    def update(self, text=None, position=None, color=None):
        """Update the text label."""
        if text is not None:
            self.text = text
        if color is not None:
            self.color = color
        if position is not None:
            self.position = position
        self.image = self.font.render(self.text, True, self.color)

    @property
    def shape(self):
        return (self.image.get_width(), self.image.get_height())

    @property
    def rect(self):
        return self.image.get_rect()

    def draw(self, surface):
        """Draw the text on the surface."""
        surface.blit(self.image, self.position)


def line_screen_edge_intersection(p, v):
    ts = []

    # vertical edges
    if not np.isclose(v[0], 0):
        ts.extend([-p[0] / v[0], (SCREEN_WIDTH - p[0]) / v[0]])

    # horizontal edges
    if not np.isclose(v[1], 0):
        ts.extend([-p[1] / v[1], (SCREEN_HEIGHT - p[1]) / v[1]])

    # return the smallest positive value
    ts = np.array(ts)
    t = np.min(ts[ts >= 0])
    return p + t * v


class Obstacle:
    def __init__(self, space, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = OBSTACLE_COLOR
        self.rect = pygame.Rect(x, y, w, h)

        # vertices of the box
        self.verts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

        self.shape = pymunk.Poly(body=space.static_body, vertices=self.verts.tolist())
        self.shape.collision_type = 0
        space.add(self.shape)

    def _compute_witness_vertices(self, point, tol=1e-3):
        right = None
        left = None
        for i in range(len(self.verts)):
            vert = self.verts[i]
            delta = vert - point
            normal = orth(delta)
            dists = (self.verts - point) @ normal
            if np.all(dists >= -tol):
                right = vert
            elif np.all(dists <= tol):
                left = vert
            if left is not None and right is not None:
                break
        return right, left

    def _compute_occlusion(self, point, tol=1e-3):
        right, left = self._compute_witness_vertices(point, tol=tol)

        delta_right = right - point
        extra_right = line_screen_edge_intersection(right, delta_right)
        normal_right = orth(delta_right)

        delta_left = left - point
        extra_left = line_screen_edge_intersection(left, delta_left)
        normal_left = orth(delta_left)

        screen_dists = []
        screen_vs = []
        for v in SCREEN_VERTS:
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
            v2 = np.sum((self.verts - point) ** 2, axis=1)
            idx = np.argmin(v2)
            if v2[idx] <= r**2:
                return unit(point - self.verts[idx, :])
        return None

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

    def draw_occlusion(self, surface, viewpoint):
        ps = self._compute_occlusion(viewpoint)
        pygame.gfxdraw.aapolygon(surface, ps, (100, 100, 100))
        pygame.gfxdraw.filled_polygon(surface, ps, (100, 100, 100))


class Entity:
    """Pymunk entity."""

    def __init__(self, space, body, shape):
        self.body = body
        self.shape = shape
        space.add(body, shape)

    @property
    def id(self):
        return self.body.id

    @property
    def position(self):
        return self.body.position

    @property
    def velocity(self):
        return self.body.velocity

    def remove(self):
        space = self.body.space
        space.remove(self.shape, self.body)


class Agent(Entity):
    def __init__(self, space, position, color):
        body = pymunk.Body(PLAYER_MASS, float("inf"))
        body.position = position

        shape = pymunk.Circle(body, 10, (0, 0))
        shape.color = color
        shape.collision_type = 1

        super().__init__(space, body, shape)

        self.health = 5
        self.ammo = CLIP_SIZE
        self.shot_cooldown = 0
        self.reload_ticks = 0
        self.target_velocity = np.zeros(2)

    def draw(self, surface):
        pygame.gfxdraw.aacircle(
            surface,
            int(self.position[0]),
            int(self.position[1]),
            int(self.shape.radius),
            self.shape.color,
        )
        pygame.gfxdraw.filled_circle(
            surface,
            int(self.position[0]),
            int(self.position[1]),
            int(self.shape.radius),
            self.shape.color,
        )

    def move(self, velocity):
        self.target_velocity += velocity

    def tick(self, obstacles):
        self.shot_cooldown = max(0, self.shot_cooldown - 1)

        # if done reloading, fill clip
        if self.reload_ticks == 1:
            self.ammo = CLIP_SIZE
        self.reload_ticks = max(0, self.reload_ticks - 1)

        v = self.target_velocity

        # don't leave the screen
        if self.position[0] >= SCREEN_WIDTH - self.shape.radius:
            v[0] = min(0, v[0])
        elif self.position[0] <= self.shape.radius:
            v[0] = max(0, v[0])
        if self.position[1] >= SCREEN_HEIGHT - self.shape.radius:
            v[1] = min(0, v[1])
        elif self.position[1] <= self.shape.radius:
            v[1] = max(0, v[1])

        # don't walk into an obstacle
        for obstacle in obstacles:
            n = obstacle.compute_collision_normal(self.position, self.shape.radius)
            if n is not None and n @ v < 0:
                t = orth(n)
                v = (t @ v) * t
        self.body.velocity = tuple(v)

        self.target_velocity = np.zeros(2)

    def reload(self):
        self.reload_ticks = RELOAD_TICKS

    def shoot(self, target):
        if self.shot_cooldown > 0 or self.reload_ticks > 0:
            return None

        norm = np.linalg.norm(target - self.position)
        if norm > 0:
            direction = (target - self.position) / norm
            self.shot_cooldown = SHOT_COOLDOWN_TICKS
            self.ammo -= 1
            if self.ammo == 0:
                self.reload()
            self.target_velocity -= 0.5 * PLAYER_VELOCITY * direction
            return Projectile(
                space=self.body.space,
                position=self.position,
                velocity=BULLET_VELOCITY * direction,
                agent_id=self.id,
            )
        else:
            return None


class Projectile(Entity):
    def __init__(self, space, position, velocity, agent_id):
        self.agent_id = agent_id

        body = pymunk.Body(BULLET_MASS, float("inf"))
        body.position = tuple(position)
        body.velocity = tuple(velocity)

        shape = pymunk.Circle(body, 3, (0, 0))
        shape.color = (0, 0, 0, 255)
        shape.collision_type = 2

        super().__init__(space, body, shape)

    def draw(self, surface):
        pygame.draw.circle(
            surface, self.shape.color, self.body.position, self.shape.radius
        )


class AgentAction:
    def __init__(self, velocity, target):
        self.velocity = velocity
        self.target = target


class Blood:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


class Game:
    def __init__(self):
        self.space = pymunk.Space()

        self.font = pygame.font.SysFont(None, 28)
        self.ammo_text = Text(
            text=f"Ammo: {CLIP_SIZE}",
            font=self.font,
            position=(20, SCREEN_HEIGHT - 60),
            color=(0, 0, 0),
        )
        self.health_text = Text(
            text="Health: 5",
            font=self.font,
            position=(20, SCREEN_HEIGHT - 30),
            color=(0, 0, 0),
        )

        self.texts = [self.ammo_text, self.health_text]

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.screen.fill(BACKGROUND_COLOR)
        self.screen_rect = self.screen.get_bounding_rect()
        pygame.display.flip()

        self.keys_down = set()

        self.projectiles = {}

        self.obstacles = [
            Obstacle(self.space, 80, 80, 80, 80),
            Obstacle(self.space, 0, 230, 160, 40),
            Obstacle(self.space, 80, 340, 80, 80),
            Obstacle(self.space, 250, 80, 40, 220),
            Obstacle(self.space, 290, 80, 100, 40),
            Obstacle(self.space, 390, 260, 110, 40),
            Obstacle(self.space, 250, 380, 200, 40),
        ]

        self.player = Agent(
            space=self.space, position=[200, 200], color=(255, 0, 0, 255)
        )
        enemies = [Agent(space=self.space, position=[200, 300], color=(0, 0, 255, 255))]

        self.agents = {enemy.body.id: enemy for enemy in enemies}
        self.agents[self.player.body.id] = self.player

        self.bloods = []

        def bullet_obstacle_handler(arbiter, space, data):
            # delete bullet when it hits an obstacle
            body_id = arbiter.shapes[1].body.id
            self.projectiles[body_id].remove()
            self.projectiles.pop(body_id)
            return False

        def bullet_agent_handler(arbiter, space, data):
            # bullet_id = arbiter.shapes[1].body.id
            # agent_id = arbiter.shapes[0].body.id
            #
            # bullet = self.projectiles[bullet_id]
            #
            # # an agent cannot be hit by its own bullet
            # if bullet.agent_id == agent_id:
            #     return False
            #
            # agent = self.agents[agent_id]
            # agent.health -= 1
            # if agent.health <= 0:
            #     print("dead!")
            #     agent.remove()
            #     self.agents.pop(agent_id)
            #
            # start = arbiter.contact_point_set.points[0].point_a
            # blood = Blood(start + 25 * unit(bullet.velocity), (0, 0))
            # self.bloods.append(blood)
            # agent.target_velocity += 0.5 * PLAYER_VELOCITY * unit(bullet.velocity)
            #
            # bullet.remove()
            # self.projectiles.pop(bullet_id)

            return False

        def obstacle_agent_handler(arbiter, space, data):
            return False

        self.space.add_collision_handler(0, 2).begin = bullet_obstacle_handler
        self.space.add_collision_handler(1, 2).begin = bullet_agent_handler
        self.space.add_collision_handler(0, 1).begin = obstacle_agent_handler

    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)

        # for blood in self.bloods:
        #     # pygame.draw.line(self.screen, (255, 0, 0), blood.p1, blood.p2, width=3)
        #     pygame.draw.circle(self.screen, (255, 0, 0), blood.p1, 2)

        for projectile in self.projectiles.values():
            projectile.draw(self.screen)

        for agent in self.agents.values():
            agent.draw(self.screen)

        for obstacle in self.obstacles:
            obstacle.draw_occlusion(self.screen, viewpoint=self.player.position)

        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        # text
        if self.player.reload_ticks > 0:
            self.ammo_text.update(text=f"Reloading...", color=(255, 0, 0))
        else:
            self.ammo_text.update(text=f"Ammo: {self.player.ammo}", color=(0, 0, 0))

        if self.player.health > 2:
            self.health_text.update(text=f"Health: {self.player.health}")
        else:
            self.health_text.update(
                text=f"Health: {self.player.health}", color=(255, 0, 0)
            )

        for text in self.texts:
            text.draw(self.screen)

        pygame.display.flip()

        # NOTE: this is how to extract the RGB values
        # raw = np.array(pygame.PixelArray(self.screen))
        # rgb = np.array([raw >> 16, raw >> 8, raw]) & 0xff
        # rgb = np.moveaxis(rgb, 0, -1)

    def step(self, actions):
        """Step the game forward in time."""
        for agent_id, agent in self.agents.items():
            if agent_id in actions:
                action = actions[agent_id]

                if action.target is not None:
                    projectile = self.player.shoot(action.target)
                    if projectile is not None:
                        self.projectiles[projectile.id] = projectile

                agent.move(action.velocity)

        # process projectiles
        projectiles_to_remove = []
        agents_to_remove = []
        for idx, projectile in self.projectiles.items():
            # projectile has left the screen
            if not self.screen_rect.collidepoint(projectile.body.position):
                projectiles_to_remove.append(idx)
                continue

            # check for collision with an agent
            s1 = projectile.position
            s2 = s1 + TIMESTEP * PHYSICS_STEP_PER_TICK * projectile.velocity
            segment = Segment(s1, s2)
            for agent_id, agent in self.agents.items():

                # agent cannot be hit by its own bullets
                if agent_id == projectile.agent_id:
                    continue

                # check for collision with the bullet's path
                # TODO segment_segment_dist might be better here
                if point_segment_dist(agent.position, segment) < agent.shape.radius:
                    projectiles_to_remove.append(idx)

                    agent.target_velocity += (
                        0.5 * PLAYER_VELOCITY * unit(projectile.velocity)
                    )
                    agent.health -= 1
                    if agent.health <= 0:
                        agents_to_remove.append(agent_id)

        for idx in projectiles_to_remove:
            self.projectiles[idx].remove()
            self.projectiles.pop(idx)

        for idx in agents_to_remove:
            print("dead!")
            self.agents[idx].remove()
            self.agents.pop(idx)

        for agent in self.agents.values():
            agent.tick(self.obstacles)

        # physics
        for _ in range(PHYSICS_STEP_PER_TICK):
            self.space.step(TIMESTEP)

    def loop(self):
        while True:

            target = None
            reload = False

            # process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    self.keys_down.add(event.key)
                elif event.type == pygame.KEYUP:
                    self.keys_down.discard(event.key)
                    reload = event.key == pygame.K_r
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    target = np.array(pygame.mouse.get_pos())

            # respond to events
            velocity = np.zeros(2)
            if pygame.K_d in self.keys_down:
                velocity[0] += 1
            if pygame.K_a in self.keys_down:
                velocity[0] -= 1
            if pygame.K_w in self.keys_down:
                velocity[1] -= 1
            if pygame.K_s in self.keys_down:
                velocity[1] += 1
            norm = np.linalg.norm(velocity)
            if norm > 0:
                velocity = PLAYER_VELOCITY * velocity / norm

            # TODO I guess reload can be dumped into the action as well
            # TODO I wonder if the action should just be the command ("go
            # left") rather than the actual velocity vector (the latter has
            # more DOFs than are actually available)
            if reload:
                self.player.reload()

            actions = {self.player.id: AgentAction(velocity, target)}

            self.step(actions)
            self.draw()
            self.clock.tick(FRAMERATE)


def main():
    pygame.init()

    game = Game()
    game.loop()


main()
