#[derive(Debug, Clone)]
pub struct SatcCircle {
    pub pos: [f64; 2],
    pub r: f64,
}

#[derive(Debug, Clone)]
pub struct SatcPolygon {
    pub pos: [f64; 2],
    pub num_points: usize,
    pub points: Vec<[f64; 2]>,
    pub angle: f64,
    pub offset: [f64; 2],
    pub calc_points: Vec<[f64; 2]>,
    pub edges: Vec<[f64; 2]>,
    pub normals: Vec<[f64; 2]>,
}

#[derive(Debug, Clone)]
pub struct SatcBox {
    pub pos: [f64; 2],
    pub w: f64,
    pub h: f64,
}

#[derive(Debug, Clone)]
pub struct SatcResponse {
    pub overlap: f64,
    pub overlap_n: [f64; 2],
    pub overlap_v: [f64; 2],
    pub a_in_b: bool,
    pub b_in_a: bool,
}

pub const SATC_TYPE_NONE: i32 = 0;
pub const SATC_TYPE_CIRCLE: i32 = 1;
pub const SATC_TYPE_POLYGON: i32 = 2;
pub const SATC_TYPE_BOX: i32 = 3;
pub const SATC_LEFT_VORONOI_REGION: i32 = -1;
pub const SATC_MIDDLE_VORONOI_REGION: i32 = 0;
pub const SATC_RIGHT_VORONOI_REGION: i32 = 1;

impl SatcCircle {
    pub fn new(pos: [f64; 2], r: f64) -> Self {
        SatcCircle { pos, r }
    }
}

impl SatcPolygon {
    pub fn new(pos: [f64; 2], points: Vec<[f64; 2]>) -> Self {
        let mut polygon = SatcPolygon {
            pos,
            num_points: 0,
            points: Vec::new(),
            angle: 0.0,
            offset: [0.0, 0.0],
            calc_points: Vec::new(),
            edges: Vec::new(),
            normals: Vec::new(),
        };
        satc_polygon_set_points(&mut polygon, points);
        polygon
    }
}

impl SatcBox {
    pub fn new(pos: [f64; 2], w: f64, h: f64) -> Self {
        SatcBox { pos, w, h }
    }
}

impl SatcResponse {
    pub fn new() -> Self {
        SatcResponse {
            overlap: f64::MAX,
            overlap_n: [0.0, 0.0],
            overlap_v: [0.0, 0.0],
            a_in_b: true,
            b_in_a: true,
        }
    }
}

impl Default for SatcResponse {
    fn default() -> Self {
        Self::new()
    }
}

fn point_dot(p: &[f64; 2], q: &[f64; 2]) -> f64 {
    p[0] * q[0] + p[1] * q[1]
}

fn point_len2(p: &[f64; 2]) -> f64 {
    point_dot(p, p)
}

fn point_len(p: &[f64; 2]) -> f64 {
    point_len2(p).sqrt()
}

fn recalculate_polygon(polygon: &mut SatcPolygon) {
    polygon.calc_points = polygon
        .points
        .iter()
        .map(|point| {
            let calculated = satc_point_add(point, &polygon.offset);
            if polygon.angle != 0.0 {
                satc_point_rotate(&calculated, polygon.angle)
            } else {
                calculated
            }
        })
        .collect();

    let mut edges = Vec::with_capacity(polygon.calc_points.len());
    let mut normals = Vec::with_capacity(polygon.calc_points.len());
    for (p1, p2) in polygon
        .calc_points
        .iter()
        .zip(polygon.calc_points.iter().cycle().skip(1))
    {
        let edge = satc_point_sub(p2, p1);
        let normal = satc_point_normalize(&satc_point_perp(&edge));
        edges.push(edge);
        normals.push(normal);
    }
    polygon.edges = edges;
    polygon.normals = normals;
}

pub fn satc_polygon_set_points(polygon: &mut SatcPolygon, points: Vec<[f64; 2]>) {
    polygon.num_points = points.len();
    polygon.points = points;
    recalculate_polygon(polygon);
}

pub fn satc_box_create(pos: [f64; 2], w: f64, h: f64) -> SatcBox {
    SatcBox::new(pos, w, h)
}

pub fn satc_box_to_polygon(box_: &SatcBox) -> SatcPolygon {
    let w = box_.w;
    let h = box_.h;
    SatcPolygon::new(box_.pos, vec![[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
}

pub fn satc_point_copy(_p: &[f64; 2], q: &[f64; 2]) -> [f64; 2] {
    *q
}

pub fn satc_point_perp(p: &[f64; 2]) -> [f64; 2] {
    [p[1], -p[0]]
}

pub fn satc_point_reverse(p: &[f64; 2]) -> [f64; 2] {
    [-p[0], -p[1]]
}

pub fn satc_point_add(p: &[f64; 2], q: &[f64; 2]) -> [f64; 2] {
    [p[0] + q[0], p[1] + q[1]]
}

pub fn satc_point_sub(p: &[f64; 2], q: &[f64; 2]) -> [f64; 2] {
    [p[0] - q[0], p[1] - q[1]]
}

pub fn satc_point_scale_xy(p: &[f64; 2], x: f64, y: f64) -> [f64; 2] {
    [p[0] * x, p[1] * y]
}

pub fn satc_point_scale_x(p: &[f64; 2], x: f64) -> [f64; 2] {
    satc_point_scale_xy(p, x, x)
}

pub fn satc_point_create(x: f64, y: f64) -> [f64; 2] {
    [x, y]
}

pub fn satc_point_rotate(p: &[f64; 2], angle: f64) -> [f64; 2] {
    let x = p[0];
    let y = p[1];
    let cos = angle.cos();
    let sin = angle.sin();
    [x * cos - y * sin, x * sin - y * cos]
}

pub fn satc_point_normalize(p: &[f64; 2]) -> [f64; 2] {
    let d = point_len(p);
    if d > 0.0 { [p[0] / d, p[1] / d] } else { *p }
}

pub fn satc_point_project(p: &[f64; 2], q: &[f64; 2]) -> [f64; 2] {
    let amt = point_dot(p, q) / point_len2(q);
    [amt * p[0], amt * p[1]]
}

pub fn satc_point_reflect(p: &[f64; 2], axis: &[f64; 2]) -> [f64; 2] {
    let projected = satc_point_scale_x(&satc_point_project(p, axis), 2.0);
    satc_point_sub(&projected, p)
}

pub fn satc_circle_create(pos: [f64; 2], r: f64) -> SatcCircle {
    SatcCircle::new(pos, r)
}

pub fn satc_circle_get_aabb(circle: &SatcCircle) -> SatcPolygon {
    let r = circle.r;
    let corner = [circle.pos[0] - r, circle.pos[1] - r];
    satc_box_to_polygon(&SatcBox::new(corner, r * 2.0, r * 2.0))
}

pub fn satc_polygon_create(pos: [f64; 2], points: Vec<[f64; 2]>) -> SatcPolygon {
    SatcPolygon::new(pos, points)
}

pub fn satc_polygon_get_aabb(polygon: &SatcPolygon) -> SatcPolygon {
    let Some(first) = polygon.points.first() else {
        return SatcPolygon::new(polygon.pos, Vec::new());
    };
    let mut x_min = first[0];
    let mut y_min = first[1];
    let mut x_max = x_min;
    let mut y_max = y_min;
    for point in polygon.points.iter().skip(1) {
        if point[0] < x_min {
            x_min = point[0];
        }
        if point[0] > x_max {
            x_max = point[0];
        }
        if point[1] < y_min {
            y_min = point[1];
        }
        if point[1] > y_max {
            y_max = point[1];
        }
    }
    let pos = [polygon.pos[0] + x_min, polygon.pos[1] + y_min];
    satc_box_to_polygon(&SatcBox::new(pos, x_max - x_min, y_max - y_min))
}

pub fn satc_polygon_get_centroid(polygon: &SatcPolygon) -> [f64; 2] {
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut ar = 0.0;
    for (p1, p2) in polygon
        .calc_points
        .iter()
        .zip(polygon.calc_points.iter().cycle().skip(1))
    {
        let a = p1[0] * p2[1] - p2[0] * p1[1];
        cx += (p1[0] + p2[0]) * a;
        cy += (p1[1] + p2[1]) * a;
        ar += a;
    }
    ar *= 3.0;
    [cx / ar, cy / ar]
}

pub fn satc_response_create() -> SatcResponse {
    SatcResponse::new()
}

pub fn satc_flatten_points_on(
    len: usize,
    points: &[[f64; 2]],
    normal: &[f64; 2],
    result: &mut [f64; 2],
) {
    let mut min = f64::MAX;
    let mut max = -f64::MAX;
    for point in points.iter().take(len) {
        let dot = point_dot(point, normal);
        if dot < min {
            min = dot;
        }
        if dot > max {
            max = dot;
        }
    }
    result[0] = min;
    result[1] = max;
}

// The argument list mirrors the C API and is fixed by the public interface.
#[allow(clippy::too_many_arguments)]
pub fn satc_is_separating_axis(
    a_pos: &[f64; 2],
    b_pos: &[f64; 2],
    a_len: usize,
    a_points: &[[f64; 2]],
    b_len: usize,
    b_points: &[[f64; 2]],
    axis: &[f64; 2],
    response: &mut SatcResponse,
) -> bool {
    let offset_v = satc_point_sub(b_pos, a_pos);
    let projected_offset = point_dot(&offset_v, axis);
    let mut range_a = [0.0, 0.0];
    let mut range_b = [0.0, 0.0];
    satc_flatten_points_on(a_len, a_points, axis, &mut range_a);
    satc_flatten_points_on(b_len, b_points, axis, &mut range_b);
    range_b[0] += projected_offset;
    range_b[1] += projected_offset;

    if range_a[0] > range_b[1] || range_b[0] > range_a[1] {
        return true;
    }

    let overlap;
    if range_a[0] < range_b[0] {
        response.a_in_b = false;
        if range_a[1] < range_b[1] {
            overlap = range_a[1] - range_b[1];
            response.b_in_a = false;
        } else {
            let option_1 = range_a[1] - range_b[0];
            let option_2 = range_b[1] - range_a[0];
            overlap = if option_1 < option_2 {
                option_1
            } else {
                -option_2
            };
        }
    } else {
        response.b_in_a = false;
        if range_a[1] > range_b[1] {
            overlap = range_a[0] - range_b[1];
            response.a_in_b = false;
        } else {
            let option_1 = range_a[1] - range_b[0];
            let option_2 = range_b[1] - range_a[0];
            overlap = if option_1 < option_2 {
                option_1
            } else {
                -option_2
            };
        }
    }

    let abs_overlap = overlap.abs();
    if abs_overlap < response.overlap {
        response.overlap = abs_overlap;
        response.overlap_n = if overlap < 0.0 {
            satc_point_reverse(axis)
        } else {
            *axis
        };
    }
    false
}

pub fn satc_voronoi_region(line: &[f64; 2], point: &[f64; 2]) -> i32 {
    let len_2 = point_len2(line);
    let dp = point_dot(point, line);
    if dp < 0.0 {
        SATC_LEFT_VORONOI_REGION
    } else if dp > len_2 {
        SATC_RIGHT_VORONOI_REGION
    } else {
        SATC_MIDDLE_VORONOI_REGION
    }
}

pub fn satc_point_in_circle(point: &[f64; 2], circle: &SatcCircle) -> bool {
    let difference_v = satc_point_sub(point, &circle.pos);
    point_len2(&difference_v) <= circle.r * circle.r
}

pub fn satc_point_in_polygon(point: &[f64; 2], polygon: &SatcPolygon) -> bool {
    let test_point = satc_box_to_polygon(&SatcBox::new(*point, 0.000001, 0.000001));
    let mut response = SatcResponse::new();
    satc_test_polygon_polygon(&test_point, polygon, &mut response) && response.a_in_b
}

pub fn satc_test_circle_circle(
    a: &SatcCircle,
    b: &SatcCircle,
    response: &mut SatcResponse,
) -> bool {
    let difference_v = satc_point_sub(&b.pos, &a.pos);
    let total_radius = a.r + b.r;
    let distance_sq = point_len2(&difference_v);
    if distance_sq > total_radius * total_radius {
        return false;
    }

    let distance = distance_sq.sqrt();
    response.overlap = total_radius - distance;
    response.overlap_n = satc_point_normalize(&difference_v);
    response.overlap_v = satc_point_scale_x(&response.overlap_n, response.overlap);
    response.a_in_b = a.r <= b.r && distance <= b.r - a.r;
    response.b_in_a = b.r <= a.r && distance <= a.r - b.r;
    true
}

pub fn satc_test_polygon_polygon(
    a: &SatcPolygon,
    b: &SatcPolygon,
    response: &mut SatcResponse,
) -> bool {
    for normal in &a.normals {
        if satc_is_separating_axis(
            &a.pos,
            &b.pos,
            a.num_points,
            &a.points,
            b.num_points,
            &b.points,
            normal,
            response,
        ) {
            return false;
        }
    }
    for normal in &b.normals {
        if satc_is_separating_axis(
            &a.pos,
            &b.pos,
            a.num_points,
            &a.points,
            b.num_points,
            &b.points,
            normal,
            response,
        ) {
            return false;
        }
    }
    response.overlap_v = satc_point_scale_x(&response.overlap_n, response.overlap);
    true
}

pub fn satc_test_polygon_circle(
    polygon: &SatcPolygon,
    circle: &SatcCircle,
    response: &mut SatcResponse,
) -> bool {
    let circle_pos = satc_point_sub(&circle.pos, &polygon.pos);
    let radius = circle.r;
    let radius2 = radius * radius;
    let points = &polygon.calc_points;
    let len = points.len();

    for i in 0..len {
        let next = if i == len - 1 { 0 } else { i + 1 };
        let prev = if i == 0 { len - 1 } else { i - 1 };
        let mut overlap = 0.0;
        let mut overlap_n = None;
        let mut edge = polygon.edges[i];
        let mut point = satc_point_sub(&circle_pos, &points[i]);

        if point_len2(&point) > radius2 {
            response.a_in_b = false;
        }

        let mut region = satc_voronoi_region(&edge, &point);
        if region == SATC_LEFT_VORONOI_REGION {
            edge = polygon.edges[prev];
            let point2 = satc_point_sub(&circle_pos, &points[prev]);
            region = satc_voronoi_region(&edge, &point2);
            if region == SATC_RIGHT_VORONOI_REGION {
                let dist = point_len(&point);
                if dist > radius {
                    return false;
                }
                response.b_in_a = false;
                overlap_n = Some(satc_point_normalize(&point));
                overlap = radius - dist;
            }
        } else if region == SATC_RIGHT_VORONOI_REGION {
            edge = polygon.edges[next];
            point = satc_point_sub(&circle_pos, &points[next]);
            region = satc_voronoi_region(&edge, &point);
            if region == SATC_LEFT_VORONOI_REGION {
                let dist = point_len(&point);
                if dist > radius {
                    return false;
                }
                response.b_in_a = false;
                overlap_n = Some(satc_point_normalize(&point));
                overlap = radius - dist;
            }
        } else {
            let normal = satc_point_normalize(&satc_point_perp(&edge));
            let dist = point_dot(&point, &normal);
            let dist_abs = dist.abs();
            if dist > 0.0 && dist_abs > radius {
                return false;
            }
            overlap_n = Some(normal);
            overlap = radius - dist;
            if dist >= 0.0 || overlap < 2.0 * radius {
                response.b_in_a = false;
            }
        }

        if let Some(normal) = overlap_n {
            if overlap.abs() < response.overlap.abs() {
                response.overlap = overlap;
                response.overlap_n = normal;
            }
        }
    }

    response.overlap_v = satc_point_scale_x(&response.overlap_n, response.overlap);
    true
}

pub fn satc_test_circle_polygon(
    circle: &SatcCircle,
    polygon: &SatcPolygon,
    response: &mut SatcResponse,
) -> bool {
    let result = satc_test_polygon_circle(polygon, circle, response);
    if result {
        let a_in_b = response.a_in_b;
        response.overlap_n = satc_point_reverse(&response.overlap_n);
        response.overlap_v = satc_point_reverse(&response.overlap_v);
        response.a_in_b = response.b_in_a;
        response.b_in_a = a_in_b;
    }
    result
}
