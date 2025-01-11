import { TagAIPolicy } from "./policy";
import { Obstacle } from "./obstacle";
import { drawCircle } from "./gui";
import { AARect, Circle, pointPolyQuery } from "./collision";
import { Vec2 } from "./math";
import { Action, Agent } from "./agent";

const TIMESTEP = 1 / 60;
const TAG_COOLDOWN = 60;
const SIZE = 50;
const SCALE = 10;

const MODEL_URL = "https://adamheins.com/projects/shadows/models";


class Treasure extends Circle {
    constructor(center, radius) {
        super(center, radius);
    }

    draw(ctx, scale=1) {
        drawCircle(ctx, this.center.scale(scale), scale * this.radius, "green");
    }

    updatePosition(width, height, obstacles) {
        const r = this.radius;
        while (true) {
            const x = (width - 2 * r) * Math.random() + r;
            const y = (height - 2 * r) * Math.random() + r;
            const p = new Vec2(x, y);

            let collision = false;
            for (let i = 0; i < obstacles.length; i++) {
                const Q = pointPolyQuery(p, obstacles[i]);
                if (Q.distance < this.radius) {
                    collision = true;
                    break;
                }
            }

            if (!collision) {
                this.center = p;
                break;
            }
        }
    }
}


class TagGame {
    constructor(width, height, scale) {
        this.width = width;
        this.height = height;
        this.scale = scale;
        this.screenRect = new AARect(0, 0, width, height);

        this.keyMap = new Map();

        document.addEventListener("keydown", (event) => {
            this.keyMap.set(event.key, true);
        });

        document.addEventListener("keyup", (event) => {
            this.keyMap.delete(event.key);
        });

        this.player = new Agent(new Vec2(10, 25), "red", false);
        this.enemy = new Agent(new Vec2(40, 25), "blue", true);
        this.agents = [this.player, this.enemy];
        this.itId = 1;

        this.obstacles = [
            new Obstacle(new Vec2(20, 27), 10, 10),
            new Obstacle(new Vec2(8, 8), 5, 5),
            new Obstacle(new Vec2(0, 37), 13, 13),
            new Obstacle(new Vec2(37, 37), 5, 5),
            new Obstacle(new Vec2(20, 8), 5, 7),
            new Obstacle(new Vec2(20, 15), 22, 5),
        ];

        this.treasures = [new Treasure(new Vec2(0, 0), 1), new Treasure(new Vec2(0, 0), 1)];
        this.treasures.forEach(treasure => treasure.updatePosition(width, height, this.obstacles));

        this.enemyPolicy = new TagAIPolicy(this.enemy, this.player, this.obstacles, this.width, this.height);

        this.tagCooldown = 0;
        this.score = 0;
        this.enemyAction = null;
    }

    draw(ctx) {
        ctx.clearRect(0, 0, this.scale * this.width, this.scale * this.height);
        this.treasures.forEach(treasure => {
            treasure.draw(ctx, this.scale);
        });
        this.agents.forEach(agent => {
            agent.draw(ctx, this.scale);
        })
        this.obstacles.forEach(obstacle => {
            obstacle.draw(ctx, this.scale);
            obstacle.drawOcclusion(ctx, this.player.position, this.screenRect, this.scale);
        });

        // draw the score
        ctx.font = "20px sans";
        ctx.fillStyle = "white";
        ctx.fillText("Score: " + this.score, this.scale * 1, this.scale * (this.height - 1));
    }

    step(dt) {
        this.tagCooldown = Math.max(0, this.tagCooldown - 1);

        // parse the keys and update the agent poses
        let lindir = 0;
        let angdir = 0;

        if (this.keyMap.has("d") || this.keyMap.has("ArrowRight")) {
            angdir -= 1;
        }
        if (this.keyMap.has("a") || this.keyMap.has("ArrowLeft")) {
            angdir += 1
        }
        if (this.keyMap.has("w") || this.keyMap.has("ArrowUp")) {
            lindir += 1
        }
        if (this.keyMap.has("s") || this.keyMap.has("ArrowDown")) {
            lindir -= 1
        }

        const lookback = this.keyMap.has("Space");
        const playerAction = new Action(new Vec2(lindir, 0), angdir, true, lookback);
        // const enemyAction = this.enemyPolicy.compute();

        // do stuff with the agents
        this.player.command(playerAction);
        if (this.enemyAction) {
            // translate from model output to actual action
            const enemyAction = new Action(new Vec2(1, 0), this.enemyAction, true, false);
            this.enemy.command(enemyAction);
        }

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

            this.obstacles.forEach(obstacle => {
                let Q = pointPolyQuery(agent.position, obstacle);
                if ((Q.distance < agent.radius) && (Q.normal.dot(v) < 0)) {
                    const tan = Q.normal.orth();
                    v = tan.scale(tan.dot(v));
                }
            });

            agent.velocity = v;
        });

        // check if someone has collected a treasure
        for (let i = 0; i < this.agents.length; i++) {
            if (i === this.itId) {
                continue;
            }

            const agent = this.agents[i];
            this.treasures.forEach(treasure => {
                const d = agent.position.subtract(treasure.center).length();
                if (d <= agent.radius + treasure.radius) {
                    if (i === 0) {
                        this.score += 1;
                    } else {
                        this.score -= 1;
                    }

                    treasure.updatePosition(this.width, this.height, this.obstacles);
                }
            });
        }

        // check if someone has been tagged
        // no tagging can happen in the cooldown period
        if (this.tagCooldown === 0) {
            const d = this.player.position.subtract(this.enemy.position).length();
            if (d < this.player.radius + this.enemy.radius) {
                this.player.it = !this.player.it;
                this.enemy.it = !this.player.it;
                this.itId = (this.itId + 1) % 2;
                this.tagCooldown = TAG_COOLDOWN;
            }
        }

        // cannot move after just being tagged
        if (this.tagCooldown > 0) {
            this.agents[this.itId].velocity = new Vec2(0, 0);
        }

        this.agents.forEach(agent => agent.step(dt));
    }
}

function main() {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    // const smallCtx = document.getElementById("small").getContext("2d");

    const scale = Math.min(canvas.width, canvas.height) / SIZE;
    console.log(scale);
    const game = new TagGame(SIZE, SIZE, scale);

    // load the AI models
    let itModelPromise = ort.InferenceSession.create(MODEL_URL + "/TagIt-v0_sac.onnx");
    let notItModelPromise = ort.InferenceSession.create(MODEL_URL + "/TagNotIt-v0_sac.onnx");

    Promise.all([itModelPromise, notItModelPromise]).then(models => {
        console.log("Loaded models.");
        const itModel = models[0];
        const notItModel = models[1];

        let lastTime = Date.now();

        function loop() {
            const now = Date.now();
            const dt = now - lastTime;
            lastTime = now;

            // Get a new action for the AI
            // from the AI's perspective, it is the agent and the player is the
            // enemy
            const agentPosition = Float32Array.from(game.enemy.position.array());
            const agentAngle = Float32Array.from([game.enemy.angle]);
            const enemyPosition = Float32Array.from(game.player.position.array());

            // treasures should be zero'd out when player is it
            let treasurePositions;
            if (game.enemy.it) {
                treasurePositions = Float32Array.from([0, 0, 0, 0]);
            } else {
                treasurePositions = Float32Array.from(game.treasures[0].center.array().concat(game.treasures[1].center.array()));
            }

            const obs = {
                agent_position: new ort.Tensor('float32', agentPosition, [1, 2]),
                agent_angle: new ort.Tensor('float32', agentAngle, [1, 1]),
                enemy_position: new ort.Tensor('float32', enemyPosition, [1, 2]),
                treasure_positions: new ort.Tensor('float32', treasurePositions, [1, 4])
            };

            let results;
            if (game.enemy.it) {
                results = itModel.run(obs);
            } else {
                results = notItModel.run(obs);
            }
            results.then(r => {
                game.enemyAction = r.tanh.cpuData[0];
                game.step(dt / 1000);
                game.draw(ctx);
            });

            // game.step(dt / 1000);
            // game.draw(ctx, SCALE);

            // game.draw(smallCtx, 1);
            // ctx.scale(0.1, 0.1);
            // const data = smallCtx.getImageData(0, 0, 50, 50);
            // let r = 0
            // for (let i = 0; i < 10000; i += 4) {
            //     if (data.data[i] === 255) {
            //         r += 1;
            //     }
            // }
            // console.log(r);
            // ctx.scale(10, 10);
            // console.log(data);

            // requestAnimationFrame(loop);
        }

        // requestAnimationFrame(loop);
        setInterval(loop, 1000 * TIMESTEP);
    });
}


window.addEventListener("load", main);
