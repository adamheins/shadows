const PLAYER_FORWARD_VEL = 75;  // px per second
const PLAYER_BACKWARD_VEL = 30;  // px per second
const PLAYER_IT_VEL = 50;  // px per second
const PLAYER_ANGVEL = 5;  // rad per second


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
    constructor(position, color, it=false) {
        this.position = position;
        this.angle = 0;
        this.color = color;
        this.radius = 3;

        this.velocity = new Vec2(0, 0);
        this.angvel = 0;

        this.it = it;
    }

    direction() {
        const c = Math.cos(this.angle);
        const s = Math.sin(this.angle);
        return new Vec2(c, -s);
    }

    draw(ctx, drawOutline=true) {
        drawCircle(ctx, this.position, this.radius, this.color);
        // const end = this.position.add(this.dir.rotate(this.angle));
        const end = this.position.add(this.direction().scale(this.radius));
        drawLine(ctx, this.position, end, "black");

        if (drawOutline && this.it) {
            drawCircle(ctx, this.position, this.radius, "yellow", false);
        }
    }

    command(action) {
        let forwardVel = 0;
        if (this.it) {
            forwardVel = PLAYER_IT_VEL;
        } else if (action.lookback) {
            forwardVel = PLAYER_BACKWARD_VEL;
        } else {
            forwardVel = PLAYER_FORWARD_VEL;
        }

        this.velocity = action.lindir.rotate(this.angle).scale(forwardVel);
        this.angvel = PLAYER_ANGVEL * action.angdir;
    }

    step(dt) {
        const dp = this.velocity.scale(dt);
        this.position = this.position.add(dp);
        this.angle = wrapToPi(this.angle + dt * this.angvel);

        this.velocity = new Vec2(0, 0);
        this.angvel = 0;
    }
}

