const TIMESTEP = 1 / 60;
const TAG_COOLDOWN = 120;
const SCALE = 10;


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
    constructor(width, height, it_model) {
        this.width = width;
        this.height = height;
        this.screenRect = new AARect(0, 0, width, height);
        this.it_model = it_model;

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
        ctx.clearRect(0, 0, SCALE * this.width, SCALE * this.height);
        this.agents.forEach(agent => {
            agent.draw(ctx, SCALE);
        })
        this.obstacles.forEach(obstacle => {
            obstacle.draw(ctx, SCALE);
            // obstacle.drawOcclusion(ctx, this.player.position, this.screenRect);
        });
        this.treasures.forEach(treasure => {
            treasure.draw(ctx, SCALE);
        });
    }

    step() {
        this.tagCooldown = Math.max(0, this.tagCooldown - 1);

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
        const playerAction = new Action(new Vec2(lindir, 0), angdir, true, lookback);
        // const enemyAction = this.enemyPolicy.compute();

        // do stuff with the agents
        this.player.command(playerAction);
        if (this.enemyAction) {
            // translate from model output to actual action
            // console.log("enemyAction = ", this.enemyAction);
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
                    console.log(this.score);

                    treasure.updatePosition(this.width, this.height, this.obstacles);
                }
            });
        }

        // check if someone has been tagged
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

        this.agents.forEach(agent => agent.step(TIMESTEP));
    }
}


async function main() {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");


    // const game = new TagGame(50, 50);
    // setInterval(() => {
    //     game.step();
    //     game.draw(ctx);
    // }, 1000 * TIMESTEP, ctx);

    try {
        const game = new TagGame(50, 50);

        const itModel = await ort.InferenceSession.create('http://localhost:8000/TagIt-v0_sac.onnx');
        const notItModel = await ort.InferenceSession.create('http://localhost:8000/TagNotIt-v0_sac.onnx');

        // let last = 0;
        // requestAnimationFrame(timestamp => {
        //     const dt = 0.001 * (timestamp - last);
        //     last = timestamp;
        //
        //     model.run(obs, action => {
        //         game.step(dt);
        //         game.draw(ctx);
        //     });
        // });

        setInterval(async function() {
            game.step();
            game.draw(ctx);

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
                results = await itModel.run(obs);
            } else {
                results = await notItModel.run(obs);
            }
            game.enemyAction = results.tanh.cpuData[0];
        }, 1000 * TIMESTEP);
    } catch (e) {
        console.log(e);
    }
}


window.addEventListener("load", function() {
    main();
});
