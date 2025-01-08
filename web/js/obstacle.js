class Obstacle extends AARect {
    constructor(position, width, height) {
        super(position.x, position.y, width, height);

        this.position = position;
        this.width = width;
        this.height = height;

        this.color = "black";
    }

    draw(ctx, scale=1) {
        drawRect(ctx, this.position.scale(scale), scale * this.width, scale * this.height, this.color);
    }

    computeWitnessVertices(point, tol=1e-8) {
        let right = null;
        let left = null;

        for (let i = 0; i < this.vertices.length; i++) {
            const vert = this.vertices[i];
            const delta = vert.subtract(point);
            const normal = delta.orth();
            const dists = this.vertices.map(v => v.subtract(point).dot(normal));

            if (dists.reduce((acc, dist) => acc && (dist >= -tol), true)) {
                right = vert;
            } else if (dists.reduce((acc, dist) => acc && (dist <= tol), true)) {
                left = vert;
            }
            if (left && right) {
                break;
            }
        }
        return [right, left];
    }

    computeOcclusion(point, screenRect) {
        const witnesses = this.computeWitnessVertices(point);
        const right = witnesses[0];
        const left = witnesses[1];

        const deltaRight = right.subtract(point).unit();
        const normalRight = deltaRight.orth();
        const distsRight = screenRect.vertices.map(v => v.subtract(point).dot(deltaRight));
        const extraRight = point.add(deltaRight.scale(Math.max(...distsRight)));

        const deltaLeft = left.subtract(point).unit();
        const normalLeft = deltaLeft.orth();
        const distsLeft = screenRect.vertices.map(v => v.subtract(point).dot(deltaLeft));
        const extraLeft = point.add(deltaLeft.scale(Math.max(...distsLeft)));

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

