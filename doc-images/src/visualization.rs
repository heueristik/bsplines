use plotters::{backend::SVGBackend, chart::ChartContext, coord::types::RangedCoordf64, prelude::*};
use plotters_arrows::TriangleArrow;

use bsplines::curve::{
    points::{ControlPoints, DataPoints, Points},
    Curve,
};

use crate::PLOTS_DIR;

const IMG_SIZE: (u32, u32) = (400, 400);
const NUM_POINTS: usize = 200;

fn linspace(start: f64, end: f64, num_points: usize) -> Vec<f64> {
    assert!(num_points > 1, "Number of points must be greater than 1");

    let step = (end - start) / (num_points - 1) as f64;

    (0..num_points).map(|i| start + i as f64 * step).collect()
}

pub struct Limits {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
}

pub fn draw_parametrized_spline_2d(
    chart_context: &mut ChartContext<SVGBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    curve: &Curve,
    derivative: usize,
    color: RGBAColor,
) {
    let point_size = 1;
    let u_values = linspace(0., 1., NUM_POINTS);
    let data = u_values.iter().cloned().map(|u| {
        let v = &curve.evaluate_derivative(u, derivative).unwrap();
        let x = v[0];
        let y = v[1];
        (x, y)
    });
    chart_context.draw_series(LineSeries::new(data, color.filled().stroke_width(point_size)).point_size(0)).unwrap();

    let mut knot_values = curve.knots.vector().data.as_vec().clone();
    knot_values.dedup();
    let knot_data = knot_values.iter().cloned().map(|u| {
        let v = &curve.evaluate_derivative(u, derivative).unwrap();
        let x = v[0];
        let y = v[1];
        (x, y)
    });

    chart_context.draw_series(knot_data.map(|point| Circle::new(point, point_size * 2, color))).unwrap();

    if curve.degree() > 1 {
        draw_arrow(chart_context, curve, derivative, color.filled(), 0.9);
    }
}

fn draw_arrow(
    chart_context: &mut ChartContext<SVGBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    curve: &Curve,
    derivative: usize,
    style: ShapeStyle,
    u: f64,
) {
    let v = &curve.evaluate_derivative(u, derivative).unwrap();
    let dv = &curve.evaluate_derivative(u, derivative + 1).unwrap();

    let x = v[0];
    let y = v[1];
    let dx = dv[0];
    let dy = dv[1];

    let norm = (dx * dx + dy * dy).sqrt();
    let dx_n = dx / norm * 0.15;
    let dy_n = dy / norm * 0.15;

    let arrow = TriangleArrow::new((x - dx_n, y - dy_n), (x, y), style).width(10).head(10);
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

pub fn draw_control_polygon_2d(
    chart_context: &mut ChartContext<SVGBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    points: &ControlPoints,
    color: RGBAColor,
) {
    let point_size = 3;
    chart_context
        .draw_series(
            points.matrix().column_iter().map(|point| Circle::new((point[0], point[1]), point_size, color.filled())),
        )
        .unwrap();

    chart_context.draw_series(LineSeries::new(points.matrix().column_iter().map(|v| (v[0], v[1])), color)).unwrap();
}

pub fn draw_data_points_2d(
    chart_context: &mut ChartContext<SVGBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    points: &DataPoints,
    connected: bool,
) {
    let point_size = 3;
    chart_context
        .draw_series(
            points.matrix().column_iter().map(|point| Circle::new((point[0], point[1]), point_size, BLACK.filled())),
        )
        .unwrap();

    if connected {
        chart_context.draw_series(LineSeries::new(points.matrix().column_iter().map(|v| (v[0], v[1])), BLACK)).unwrap();
    }
}

pub fn generate_2d_plot(filename: &str, splines: Vec<(&Curve, RGBAColor)>, limits: &Limits, data: Option<&DataPoints>) {
    for (c, _) in &splines {
        assert_eq!(c.dimension(), 2);
    }

    let mut path = String::from(PLOTS_DIR);
    path.push_str(filename);
    let area = SVGBackend::new(&path, IMG_SIZE).into_drawing_area();
    area.fill(&RGBAColor(255, 255, 255, 0.81)).unwrap(); // Matches the font color in docs.rs dark mode

    let mut chart_builder = ChartBuilder::on(&area);
    chart_builder.margin(10).set_left_and_bottom_label_area_size(20);

    let mut chart_context =
        chart_builder.build_cartesian_2d(limits.min[0]..limits.max[0], limits.min[1]..limits.max[1]).unwrap();
    chart_context.configure_mesh().draw().unwrap();

    if let Some(dp) = data {
        assert_eq!(dp.dimension(), 2);

        draw_data_points_2d(&mut chart_context, &dp, false)
    }

    for (c, color) in splines {
        draw_control_polygon_2d(&mut chart_context, &c.points, color);
        draw_parametrized_spline_2d(&mut chart_context, c, 0, color);
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
