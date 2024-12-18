// Action for the agent to take
class Action {
    constructor(lindir, angdir = 0, localFrame = true, lookback = false) {
        this.lindir = lindir;
        this.angdir = angdir;
        this.localFrame = localFrame;
        this.lookback = lookback;
    }
}

class Agent {
    constructor(position, color) {
        this.position = position;
        this.angle = 0;
        this.color = color;
        this.radius = 3;
        this.dir = new Vec2(this.radius, 0);

        this.velocity = new Vec2(0, 0);
        this.angvel = 0;
    }

    draw(ctx) {
        drawCircle(ctx, this.position, this.radius, this.color);
        const end = this.position.add(this.dir.rotate(this.angle));
        drawLine(ctx, this.position, end, "black");
    }

    command(action) {
        this.velocity = action.lindir.scale(75).rotate(this.angle);
        this.angvel = 5 * action.angdir;
    }

    step(dt) {
        const dp = this.velocity.scale(dt);
        this.position = this.position.add(dp);
        this.angle += dt * this.angvel;

        this.velocity = new Vec2(0, 0);
        this.angvel = 0;
    }


}

class Obstacle extends AARect {
    constructor(position, width, height) {
        super(position.x, position.y, width, height);

        this.position = position;
        this.width = width;
        this.height = height;

        this.color = "black";
    }

    draw(ctx) {
        drawRect(ctx, this.position, this.width, this.height, this.color);
    }

    computeWitnessVertices(point) {
        // the normal can be computed with any point in the obstacle
        const normal = this.vertices[0].subtract(point).orth();
        const dists = this.vertices.map(v => v.subtract(point).dot(normal));

        let minIdx = 0;
        let maxIdx = 0;
        for (let i = 1; i < this.vertices.length; i++) {
            if (dists[i] < dists[minIdx]) {
                minIdx = i;
            }
            if (dists[i] > dists[maxIdx]) {
                maxIdx = i;
            }
        }

        const left = this.vertices[maxIdx];
        const right = this.vertices[minIdx];
        return [right, left];
    }

    computeOcclusion(point, screenRect) {
        const witnesses = this.computeWitnessVertices(point);
        const right = witnesses[0];
        const left = witnesses[1];

        const deltaRight = right.subtract(point);
        const extraRight = lineRectEdgeIntersection(right, deltaRight, screenRect);
        const normalRight = deltaRight.orth();

        const deltaLeft = left.subtract(point);
        const extraLeft = lineRectEdgeIntersection(left, deltaLeft, screenRect);
        const normalLeft = deltaLeft.orth();

        let screenDists = [];
        let screenVs = [];
        for (let i = 0; i < screenRect.vertices.length; i++) {
            const v = screenRect.vertices[i];
            if (-v.subtract(point).dot(normalLeft) < 0) {
                continue;
            }
            const dist = v.subtract(point).dot(normalRight);
            if (dist >= 0) {
                if (screenDists.length > 0 && screenDists[0] > dist) {
                    screenVs = [v, screenVs[0]];
                    break;
                } else {
                    screenDists.push(dist);
                    screenVs.push(v);
                }
            }
        }
        return [right, extraRight].concat(screenVs).concat([extraLeft, left]);
    }

    drawOcclusion(ctx, viewpoint, screenRect) {
        const vertices = this.computeOcclusion(viewpoint, screenRect);
        drawPolygon(ctx, vertices, this.color);
    }
}

class TagGame {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.screenRect = new AARect(0, 0, width, height);

        this.keyMap = new Map();

        document.addEventListener("keydown", (event) => {
            this.keyMap.set(event.key, true);
        });

        document.addEventListener("keyup", (event) => {
            this.keyMap.delete(event.key);
        });

        this.player = new Agent(new Vec2(10, 10), "red");
        this.enemy = new Agent(new Vec2(40, 40), "blue");
        this.agents = [this.player, this.enemy];

        this.obstacles = [new Obstacle(new Vec2(20, 20), 10, 10)];
    }

    draw(ctx) {
        ctx.clearRect(0, 0, this.width, this.height);
        this.agents.forEach(agent => {
            agent.draw(ctx);
        })
        this.obstacles.forEach(obstacle => {
            obstacle.draw(ctx);
            obstacle.drawOcclusion(ctx, this.player.position, this.screenRect);
        });
    }

    step() {
        // parse the keys and update the agent poses
        let lindir = 0;
        let angdir = 0;

        if (this.keyMap.has("d")) {
            angdir -= 1;
        }
        if (this.keyMap.has("a")) {
            angdir += 1
        }
        if (this.keyMap.has("w")) {
            lindir += 1
        }
        if (this.keyMap.has("s")) {
            lindir -= 1
        }

        const lookback = this.keyMap.has("Space");
        const action = new Action(new Vec2(lindir, 0), angdir, true, lookback);

        // do stuff with the agents
        this.player.command(action);

        this.agents.forEach(agent => {
            let v = agent.velocity;

            // don't leave the screen
            if (agent.position.x >= this.width - agent.radius) {
                v.x = Math.min(0, v.x);
            } else if (agent.position.x <= agent.radius) {
                v.x = Math.max(0, v.x);
            }
            if (agent.position.y >= this.height - agent.radius) {
                v.y = Math.min(0, v.y);
            } else if (agent.position.y <= agent.radius) {
                v.y = Math.max(0, v.y);
            }

            let normal = null;
            this.obstacles.forEach(obstacle => {
                let Q = pointPolyQuery(agent.position, obstacle);
                if (Q.distance < agent.radius) {
                    normal = Q.normal;
                }
            });
            if (normal && normal.dot(v) < 0) {
                const tan = normal.orth();
                v = tan.scale(tan.dot(v));
            }

            agent.velocity = v;
        });

        this.player.step(1 / 60);
    }
}


function main() {

    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    const game = new TagGame(50, 50);

    setInterval(() => {
        game.step();
        game.draw(ctx);
    }, 1000 / 60, ctx);
}


window.addEventListener("load", function() {
    main();
})


