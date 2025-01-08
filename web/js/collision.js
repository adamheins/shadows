class CollisionQuery {
    constructor(distance=null, time=null, normal=null, p1=null, p2=null, intersect=null
    ) {
        this.p1 = p1;
        this.p2 = p2;
        this.normal = normal;
        this.distance = distance;
        this.time = time;
        this.intersect = intersect;
    }
}

class Circle {
    constructor(center, radius) {
        this.center = center;
        this.radius = radius;
    }
}

class Segment {
    constructor(start, end) {
        this.start = start;
        this.end = end;

        this.v = this.end.subtract(start);
        this.direction = this.v.unit();
        this.normal = this.direction.orth();
    }
}

class Polygon {
    constructor(vertices) {
        this.vertices = vertices;

        this.edges = [];
        for (let i = 0; i < this.vertices.length - 1; i++) {
            this.edges.push(new Segment(this.vertices[i], this.vertices[i + 1]));
        }
        this.edges.push(new Segment(this.vertices[this.vertices.length - 1], this.vertices[0]));

        this.in_normals = this.edges.map(edge => edge.normal);
        this.out_normals = this.in_normals.map(n => n.scale(-1));
    }
}

class AARect extends Polygon {
    constructor(x, y, w, h) {
        const vertices = [new Vec2(x, y), new Vec2(x, y + h), new Vec2(x + w, y + h), new Vec2(x + w, y)];
        super(vertices);

        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
    }
}


function lineRectEdgeIntersection(p, v, rect, tol = 1e-8) {
    let ts = [];

    // vertical edges
    if (Math.abs(v.x) > tol) {
        ts = ts.concat([(rect.x - p.x) / v.x, (rect.x + rect.w - p.x) / v.x]);
    }

    // horizontal edges
    if (Math.abs(v.y) > tol) {
        ts = ts.concat([(rect.y - p.y) / v.y, (rect.y + rect.h - p.y) / v.y]);
    }

    // return the smallest positive value
    const t = Math.min(...ts.filter(t => t >= 0));
    return p.add(v.scale(t));
}


function pointSegmentQuery(point, segment, tol = 1e-8) {
    const q = segment.start.subtract(point);

    const t = -q.dot(segment.v) / segment.v.dot(segment.v);
    if (t >= 0 && t <= 1) {
        const r = segment.start.add(segment.v.scale(t));
        const d = point.subtract(r).length();
        const intersect = Math.abs(d) < tol;
        return new CollisionQuery(d, null, null, point, r, intersect);
    }

    const d1 = point.subtract(segment.start).length();
    const d2 = point.subtract(segment.end).length();
    if (d1 < d2) {
        let normal = point.subtract(segment.start).unit();
        return new CollisionQuery(d1, null, normal, point, segment.start, false);
    }
    let normal = point.subtract(segment.end).unit();
    return new CollisionQuery(d2, null, normal, point, segment.end, false);
}


function pointPolyQuery(point, poly) {
    const n = poly.vertices.length;

    // inward-facing depth values for each edge
    let depths = [];
    for (let i = 0; i < n; i++) {
        const v = poly.vertices[i];
        const normal = poly.in_normals[i];
        depths.push(normal.dot(point.subtract(v)));
    }

    let minIdx = 0;
    for (let i = 1; i < n; i++) {
        if (depths[i] < depths[minIdx]) {
            minIdx = i;
        }
    }

    if (depths[minIdx] >= 0) {
        const normal = poly.out_normals[minIdx];
        return new CollisionQuery(0, null, normal, point, point, true);
    }

    // detect if a vertex is the closest point
    // we need only check the vertices on the closest edge
    const prev_idx = (minIdx - 1) % n;
    const next_idx = (minIdx + 1) % n;
    if (depths[prev_idx] < 0 || depths[next_idx] < 0) {
        let v = null;
        if (depths[prev_idx] < 0) {
            v = poly.vertices[minIdx];
        } else {
            v = poly.vertices[next_idx];
        }
        const dist = point.subtract(v).length();
        const normal = point.subtract(v).unit();
        return new CollisionQuery(dist, null, normal, point, v, false);
    }

    // otherwise we know the closest point lies on the segment
    let Q = pointSegmentQuery(point, poly.edges[minIdx]);
    Q.normal = poly.out_normals[minIdx];
    return Q;
}
