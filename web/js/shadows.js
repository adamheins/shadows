function drawCircle(ctx, position, radius, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(position.x, position.y, radius, 0, 2 * Math.PI);
    ctx.fill();
}

function drawPolygon(ctx, vertices, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(vertices[0].x, vertices[1].y);
    for (let i = 1; i < vertices.length; i++) {
        ctx.lineTo(vertices[i].x, vertices[i].y);
    }
    ctx.closePath();
    ctx.fill();
}

function drawRect(ctx, position, width, height, color) {
    ctx.fillStyle = color;
    ctx.fillRect(position.x, position.y, width, height);
}

function drawLine(ctx, start, end, color) {
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
}

// Action for the agent to take
class Action {
    constructor(lindir, angdir = 0, localFrame = true, lookback = false) {
        this.lindir = lindir;
        this.angdir = angdir;
        this.localFrame = localFrame;
        this.lookback = lookback;
    }
}

class Vec2 {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    scale(s) {
        return new Vec2(s * this.x, s * this.y);
    }

    dot(other) {

    }

    add(other) {
        return new Vec2(this.x + other.x, this.y + other.y);
    }

    rotate(angle) {
        const s = Math.sin(angle);
        const c = Math.cos(angle);
        const x = c * this.x + s * this.y;
        const y = -s * this.x + c * this.y;
        return new Vec2(x, y);
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

class Obstacle {
    constructor(position, width, height) {
        this.position = position;
        this.width = width;
        this.height = height;

        this.color = "black";
    }

    draw(ctx) {
        drawRect(ctx, this.position, this.width, this.height, this.color);
    }

    computeOcclusion(point) {

    }

    drawOcclusion(ctx, viewpoint) {

    }
}

class Game {
    constructor(width, height) {
        this.width = width;
        this.height = height;

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

            agent.velocity = v;
        });

        this.player.step(1 / 60);
    }
}


function main() {

    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    const game = new Game(50, 50);

    setInterval(() => {
        game.step();
        game.draw(ctx);
    }, 1000 / 60, ctx);
}


window.addEventListener("load", function() {
    main();
})


