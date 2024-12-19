
// Action for the agent to take
class Action {
    constructor(lindir, angdir = 0, localFrame = true, lookback = false) {
        this.lindir = lindir;
        this.angdir = angdir;
        this.localFrame = localFrame;
        this.lookback = lookback;
    }
}


class TagAIPolicy {
    constructor(agent, player, obstacles, width, height) {
        this.agent = agent;
        this.player = player;
        this.obstacles = obstacles;

        this.width = width;
        this.height = height;
        this.screenCenter = new Vec2(0.5 * width, 0.5 * height);
    }

    itPolicy() {
        const r = this.player.position.subtract(this.agent.position);

        // steer toward the player
        const a = angle2pi(r, this.agent.angle)
        let angvel = 0;
        if (a < Math.PI) {
            angvel = 1;
        } else if (a > Math.PI) {
            angvel = -1;
        } else {
            angvel = 0;
        }
        return new Action(new Vec2(1, 0), angvel);
    }

    notItPolicy() {
        const r = this.player.position.subtract(this.agent.position);
        const d = this.agent.direction();

        let angvel = 0;
        if (d.dot(r) < 0) {
            // we are already facing away from the player, so take whichever
            // direction orthogonal to center point moves us farther away from
            // the player
            const p = this.agent.position.subtract(this.screenCenter);
            let v = p.orth();
            if (v.dot(r) > 0) {
                v = v.scale(-1);
            }
            const a = angle2pi(v, this.agent.angle)
            if (a < Math.PI) {
                angvel = 1;
            } else if (a > Math.PI) {
                angvel = -1;
            }
        } else {
            // steer away from the player
            const a = angle2pi(r, this.agent.angle)
            if (a < Math.PI) {
                angvel = -1;
            } else if (a > Math.PI) {
                angvel = 1;
            }
        }
        return new Action(new Vec2(1, 0), angvel);
    }

    compute() {
        if (this.agent.it) {
            return this.itPolicy();
        }
        return this.notItPolicy();
    }
}

