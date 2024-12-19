const TIMESTEP = 1 / 60;
const TAG_COOLDOWN = 120;


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

        this.player = new Agent(new Vec2(10, 10), "red", false);
        this.enemy = new Agent(new Vec2(40, 40), "blue", true);
        this.agents = [this.player, this.enemy];
        this.itId = 1;

        this.obstacles = [new Obstacle(new Vec2(20, 20), 10, 10)];

        this.enemyPolicy = new TagAIPolicy(this.enemy, this.player, this.obstacles, this.width, this.height);

        this.tagCooldown = 0;
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
        const enemyAction = this.enemyPolicy.compute();

        // do stuff with the agents
        this.player.command(playerAction);
        this.enemy.command(enemyAction);

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

async function loadModel() {
    const session = await ort.InferenceSession.create('http://localhost:8000/dqn.onnx');
    return session;
}


function main() {

    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    // const model = loadModel();
    const game = new TagGame(50, 50);

    // let model = null;
    // ort.InferenceSession.create('http://localhost:8000/dqn.onnx').then(session => {
    //     model = session;
    // }).catch(e => {
    //     console.log(e);
    // });

    setInterval(() => {
        game.step();
        game.draw(ctx);
    }, 1000 * TIMESTEP, ctx);
}


window.addEventListener("load", function() {
    main();
})


