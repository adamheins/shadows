class Vec2 {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    scale(s) {
        return new Vec2(s * this.x, s * this.y);
    }

    dot(other) {
        return this.x * other.x + this.y * other.y;
    }

    add(other) {
        return new Vec2(this.x + other.x, this.y + other.y);
    }

    subtract(other) {
        return new Vec2(this.x - other.x, this.y - other.y);
    }

    rotate(angle) {
        const s = Math.sin(angle);
        const c = Math.cos(angle);
        const x = c * this.x + s * this.y;
        const y = -s * this.x + c * this.y;
        return new Vec2(x, y);
    }

    orth() {
        return new Vec2(this.y, -this.x);
    }

    length() {
        return Math.sqrt(this.dot(this));
    }

    unit() {
        const d = this.length();
        if (d > 0) {
            return this.scale(1 / d);
        }
        return this;
    }
}
