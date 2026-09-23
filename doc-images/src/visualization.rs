use plotters::{
    backend::SVGBackend,
    chart::ChartContext,
    coord::{CoordTranslate, Shift, cartesian::Cartesian3d, types::RangedCoordf64},
    prelude::*,
    style::text_anchor::{HPos, Pos, VPos},
};
use plotters_arrows::TriangleArrow;

use bsplines::{ControlPoints, Curve, DataPoints, Points};

use crate::PLOTS_DIR;

const IMG_SIZE: (u32, u32) = (400, 400);
/// The plot background. On the dark theme of docs.rs, it matches the font color.
const BACKGROUND: RGBAColor = RGBAColor(255, 255, 255, 0.81);
const NUM_POINTS: usize = 200;

/// The distance in pixels from the middle of an axis to its name.
const AXIS_NAME_DISTANCE: f64 = 40.0;

fn linspace(start: f64, end: f64, num_points: usize) -> Vec<f64> {
    assert!(num_points > 1, "Number of points must be greater than 1");

    let step = (end - start) / (num_points - 1) as f64;

    (0..num_points).map(|i| start + i as f64 * step).collect()
}

pub struct Limits {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
}

/// Converts a point into a coordinate of a 2D or 3D chart.
pub trait Coordinate: Clone + 'static {
    /// The length of the direction arrow on a curve, in axis units.
    const ARROW_LENGTH: f64;

    fn from_point(point: &[f64]) -> Self;
}

impl Coordinate for (f64, f64) {
    const ARROW_LENGTH: f64 = 0.15;

    fn from_point(point: &[f64]) -> Self {
        (point[0], point[1])
    }
}

/// The second axis of a 3D chart is vertical, so the point (x, y, z) becomes (y, z, x).
/// This order keeps z vertical and the axes right-handed.
impl Coordinate for (f64, f64, f64) {
    // A unit is shorter in the 3D chart than in the 2D chart.
    const ARROW_LENGTH: f64 = 0.25;

    fn from_point(point: &[f64]) -> Self {
        (point[1], point[2], point[0])
    }
}

pub fn draw_parametrized_spline<CT: CoordTranslate<From: Coordinate>>(
    chart_context: &mut ChartContext<SVGBackend, CT>,
    curve: &Curve,
    derivative: usize,
    color: RGBAColor,
) {
    let point_size = 1;
    let u_values = linspace(0., 1., NUM_POINTS);
    let data = u_values.iter().cloned().map(|u| {
        let v = &curve.evaluate_derivative(u, derivative).unwrap();
        CT::From::from_point(v.as_slice())
    });
    chart_context.draw_series(LineSeries::new(data, color.filled().stroke_width(point_size)).point_size(0)).unwrap();

    let mut knot_values = curve.knots().vector().data.as_vec().clone();
    knot_values.dedup();
    let knot_data = knot_values.iter().cloned().map(|u| {
        let v = &curve.evaluate_derivative(u, derivative).unwrap();
        CT::From::from_point(v.as_slice())
    });

    chart_context.draw_series(knot_data.map(|point| Circle::new(point, point_size * 2, color))).unwrap();

    if curve.degree() > 1 {
        draw_arrow(chart_context, curve, derivative, color.filled(), 0.9);
    }
}

fn draw_arrow<CT: CoordTranslate<From: Coordinate>>(
    chart_context: &mut ChartContext<SVGBackend, CT>,
    curve: &Curve,
    derivative: usize,
    style: ShapeStyle,
    u: f64,
) {
    let v = &curve.evaluate_derivative(u, derivative).unwrap();
    let dv = &curve.evaluate_derivative(u, derivative + 1).unwrap();

    let nock = v - dv / dv.norm() * CT::From::ARROW_LENGTH;

    let arrow = TriangleArrow::new(CT::From::from_point(nock.as_slice()), CT::From::from_point(v.as_slice()), style)
        .width(10)
        .head(10);
    chart_context.plotting_area().draw(&arrow).unwrap();
}

pub fn draw_spline_1d(
    chart_context: &mut ChartContext<SVGBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    curve: &Curve,
    derivative: usize,
    style: ShapeStyle,
    dim: usize,
) {
    let u_values = linspace(0., 1., NUM_POINTS);

    let data = u_values.iter().cloned().map(|u| {
        let v = &curve.evaluate_derivative(u, derivative).unwrap();
        (u, v[dim])
    });
    chart_context.draw_series(LineSeries::new(data, style).point_size(1)).unwrap();
}

pub fn draw_control_polygon<CT: CoordTranslate<From: Coordinate>>(
    chart_context: &mut ChartContext<SVGBackend, CT>,
    points: &ControlPoints,
    color: RGBAColor,
) {
    let point_size = 3;
    chart_context
        .draw_series(
            points
                .matrix()
                .column_iter()
                .map(|point| Circle::new(CT::From::from_point(point.as_slice()), point_size, color.filled())),
        )
        .unwrap();

    chart_context
        .draw_series(LineSeries::new(points.matrix().column_iter().map(|v| CT::From::from_point(v.as_slice())), color))
        .unwrap();
}

pub fn draw_data_points<CT: CoordTranslate<From: Coordinate>>(
    chart_context: &mut ChartContext<SVGBackend, CT>,
    points: &DataPoints,
    connected: bool,
) {
    let point_size = 3;
    chart_context
        .draw_series(
            points
                .matrix()
                .column_iter()
                .map(|point| Circle::new(CT::From::from_point(point.as_slice()), point_size, BLACK.filled())),
        )
        .unwrap();

    if connected {
        chart_context
            .draw_series(LineSeries::new(
                points.matrix().column_iter().map(|v| CT::From::from_point(v.as_slice())),
                BLACK,
            ))
            .unwrap();
    }
}

/// Writes the axis names outside the tick labels. In the view of `generate_3d_plot`, plotters writes the tick
/// labels of x and y below the front edges of the floor, and the tick labels of z beside the left edge.
fn draw_axis_names(
    area: &DrawingArea<SVGBackend, Shift>,
    chart_context: &ChartContext<SVGBackend, Cartesian3d<RangedCoordf64, RangedCoordf64, RangedCoordf64>>,
    limits: &Limits,
) {
    let (min, max) = (&limits.min, &limits.max);
    let middle = |axis: usize| (min[axis] + max[axis]) / 2.0;
    let pixel = |point: [f64; 3]| chart_context.as_coord_spec().translate(&Coordinate::from_point(&point));
    let center = pixel([middle(0), middle(1), middle(2)]);
    let style =
        TextStyle::from(("sans-serif", 16, FontStyle::Italic).into_font()).pos(Pos::new(HPos::Center, VPos::Center));

    for (name, axis_middle) in
        [("x", [middle(0), max[1], min[2]]), ("y", [max[0], middle(1), min[2]]), ("z", [max[0], min[1], middle(2)])]
    {
        let (x, y) = pixel(axis_middle);
        let (dx, dy) = ((x - center.0) as f64, (y - center.1) as f64);
        let scale = AXIS_NAME_DISTANCE / dx.hypot(dy);
        area.draw(&Text::new(name, (x + (dx * scale) as i32, y + (dy * scale) as i32), style.clone())).unwrap();
    }
}

pub fn generate_2d_plot(filename: &str, splines: Vec<(&Curve, RGBAColor)>, limits: &Limits, data: Option<&DataPoints>) {
    for (c, _) in &splines {
        assert_eq!(c.dimension(), 2);
    }

    let mut path = String::from(PLOTS_DIR);
    path.push_str(filename);
    let area = SVGBackend::new(&path, IMG_SIZE).into_drawing_area();
    area.fill(&BACKGROUND).unwrap();

    let mut chart_builder = ChartBuilder::on(&area);
    chart_builder.margin(10).set_left_and_bottom_label_area_size(20);

    let mut chart_context =
        chart_builder.build_cartesian_2d(limits.min[0]..limits.max[0], limits.min[1]..limits.max[1]).unwrap();
    chart_context.configure_mesh().draw().unwrap();

    if let Some(dp) = data {
        assert_eq!(dp.dimension(), 2);

        draw_data_points(&mut chart_context, dp, false)
    }

    for (c, color) in splines {
        draw_control_polygon(&mut chart_context, c.control_points(), color);
        draw_parametrized_spline(&mut chart_context, c, 0, color);
    }

    area.present()
        .expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", path);
}

pub fn generate_3d_plot(filename: &str, splines: Vec<(&Curve, RGBAColor)>, limits: &Limits, data: Option<&DataPoints>) {
    for (c, _) in &splines {
        assert_eq!(c.dimension(), 3);
    }

    let mut path = String::from(PLOTS_DIR);
    path.push_str(filename);
    let area = SVGBackend::new(&path, IMG_SIZE).into_drawing_area();
    area.fill(&BACKGROUND).unwrap();

    let mut chart_builder = ChartBuilder::on(&area);
    // The wide left margin leaves space for the name of the z axis.
    chart_builder.margin(10).margin_left(30);

    let (min, max) = (<(f64, f64, f64)>::from_point(&limits.min), <(f64, f64, f64)>::from_point(&limits.max));
    let mut chart_context = chart_builder.build_cartesian_3d(min.0..max.0, min.1..max.1, min.2..max.2).unwrap();
    // The axis names need this view. With a larger yaw or pitch, plotters can move the tick labels to other edges.
    chart_context.with_projection(|mut projection| {
        projection.yaw = 0.7;
        projection.pitch = 0.5;
        projection.scale = 0.75;
        projection.into_matrix()
    });
    chart_context
        .configure_axes()
        .axis_panel_style(TRANSPARENT)
        .light_grid_style(BLACK.mix(0.1))
        .max_light_lines(2)
        .draw()
        .unwrap();
    draw_axis_names(&area, &chart_context, limits);

    if let Some(dp) = data {
        assert_eq!(dp.dimension(), 3);

        draw_data_points(&mut chart_context, dp, false)
    }

    for (c, color) in splines {
        draw_control_polygon(&mut chart_context, c.control_points(), color);
        draw_parametrized_spline(&mut chart_context, c, 0, color);
    }

    area.present()
        .expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", path);
}

pub fn generate_1d_plot(filename: &str, splines: Vec<(&Curve, RGBAColor)>, limits: &Limits) {
    for (c, _) in &splines {
        assert_eq!(c.dimension(), 1);
    }

    let mut path = String::from(PLOTS_DIR);
    path.push_str(filename);
    let area = SVGBackend::new(&path, IMG_SIZE).into_drawing_area();
    area.fill(&BACKGROUND).unwrap();

    let mut chart_builder = ChartBuilder::on(&area);
    chart_builder.margin(10).set_left_and_bottom_label_area_size(20);

    let mut chart_context = chart_builder.build_cartesian_2d(0.0..1.0, limits.min[0]..limits.max[0]).unwrap();
    chart_context.configure_mesh().draw().unwrap();

    for (bs, color) in splines {
        draw_spline_1d(&mut chart_context, bs, 0, color.filled(), 0);
    }

    area.present()
        .expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", path);
}
