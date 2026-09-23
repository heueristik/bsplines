use std::ops::AddAssign;

use nalgebra::{dmatrix, dvector};
use plotters::{prelude::*, style::full_palette::TEAL};

use bsplines::{Constraints, ControlPoints, Curve, DataPoints, Points};

use crate::visualization::Limits;

mod visualization;

const PLOTS_DIR: &str = "doc-images/plots/";

const RED_100: RGBAColor = RGBAColor(255, 0, 0, 1.0);
const BLUE_100: RGBAColor = RGBAColor(0, 0, 255, 1.0);

const PURPLE_100: RGBAColor = RGBAColor(200, 0, 200, 1.0);

fn limits() -> Limits {
    Limits { min: vec![-3.0, -3.0], max: vec![3.0, 3.0] }
}

fn scattered_data_points() -> DataPoints {
    DataPoints::new(dmatrix![
        -2.50,-2.45,-2.15,-1.70,-1.50,-1.35,-1.20, 0.05, 0.20, 0.55, 0.65, 1.00, 1.20, 1.50, 1.75, 2.00, 2.15, 2.50;
        -2.55,-2.10,-2.45,-2.60,-2.15,-2.15,-1.85,-1.20,-0.70,-0.90,-0.20, 2.00, 0.95, 1.40,-0.70,-1.90,-1.70,-2.15;
    ])
}

fn example_spline(p: usize) -> Curve {
    Curve::with_uniform_knots(
        ControlPoints::new(dmatrix![
            -2.5,-1.5,-0.5, 1.0, 2.0, 0.0;
            -2.5, 1.0,-1.5,-2.0, 1.0, 2.0;
        ]),
        p,
    )
    .unwrap()
}

fn fit_plots() {
    let p = 2;
    let dp = scattered_data_points();
    let bs_max = Curve::fit(&dp, p).loose_ends().build().unwrap();

    let bs_half = Curve::fit(&dp, p).polygon_segments(dp.count() / 3).loose_ends().build().unwrap();

    let bs_half_penalized =
        Curve::fit(&dp, p).polygon_segments(dp.count() / 3).loose_ends().penalized(0.5, 2).build().unwrap();

    visualization::generate_2d_plot("generation/points.svg", vec![], &limits(), Some(&dp));
    visualization::generate_2d_plot("generation/fit-loose-all.svg", vec![(&bs_max, RED_100)], &limits(), Some(&dp));
    visualization::generate_2d_plot("generation/fit-loose-half.svg", vec![(&bs_half, RED_100)], &limits(), Some(&dp));
    visualization::generate_2d_plot(
        "generation/fit-loose-half-penalized.svg",
        vec![(&bs_half_penalized, RED_100)],
        &limits(),
        Some(&dp),
    );
}

fn interpolation_plot() {
    let dp = scattered_data_points();
    let c = Curve::interpolate(&dp, 2).unwrap();

    visualization::generate_2d_plot("generation/interpolation.svg", vec![(&c, RED_100)], &limits(), Some(&dp));
}

fn manual_plot() {
    let dp = scattered_data_points();
    let c = Curve::with_uniform_knots(ControlPoints::new(dp.matrix().clone()), 2).unwrap();
    visualization::generate_2d_plot("generation/manual.svg", vec![(&c, RED_100)], &limits(), Some(&dp));
}

fn derivatives_plot() {
    let bs_k0 = Curve::with_uniform_knots(
        ControlPoints::new(dmatrix![
            -0.25, -0.05, 0.0, 0.05, 0.25;
        ]),
        3,
    )
    .unwrap();

    let lim = Limits { min: vec![-4.4], max: vec![9.0] };

    let bs_k1 = bs_k0.derivative_curve(1).unwrap();
    let bs_k2 = bs_k0.derivative_curve(2).unwrap();
    let bs_k3 = bs_k0.derivative_curve(3).unwrap();

    visualization::generate_1d_plot(
        "derivatives.svg",
        vec![(&bs_k0, RED_100), (&bs_k1, PURPLE_100), (&bs_k2, BLUE_100), (&bs_k3, TEAL.to_rgba())],
        &lim,
    );
}

fn insert_plots() {
    let p = 2;
    let mut c = example_spline(p);
    let lim = limits();
    let u = 0.8;
    visualization::generate_2d_plot("manipulation/insert-before.svg", vec![(&c, RED_100)], &lim, None);

    c.insert_knot(u).unwrap();

    visualization::generate_2d_plot("manipulation/insert-after.svg", vec![(&c, BLUE_100)], &lim, None);
}

fn split_plots() {
    let c = example_spline(2);
    let lim = limits();
    visualization::generate_2d_plot("manipulation/split-before.svg", vec![(&c, PURPLE_100)], &lim, None);
    let (a, b) = c.split(0.5).unwrap();
    visualization::generate_2d_plot("manipulation/split-after.svg", vec![(&a, RED_100), (&b, BLUE_100)], &lim, None);
}

fn reverse_plots() {
    let mut c = example_spline(2);
    let lim = limits();
    visualization::generate_2d_plot("manipulation/reverse-before.svg", vec![(&c, RED_100)], &lim, None);
    c.reverse();
    visualization::generate_2d_plot("manipulation/reverse-after.svg", vec![(&c, PURPLE_100)], &lim, None);
}

fn merge_plots() {
    let c = example_spline(2);
    let (l_unshifted, r_unshifted) = c.split(0.5).unwrap();

    let mut points_a = l_unshifted.control_points().matrix().clone();
    let mut points_b = r_unshifted.control_points().matrix().clone();

    // Shift points
    points_a.column_mut(points_a.ncols() - 1).add_assign(dvector![0.25, -0.25]);
    points_b.column_mut(0).add_assign(dvector![0.25, 0.25]);

    let a = Curve::new(l_unshifted.knots().clone(), ControlPoints::new(points_a)).unwrap();
    let b = Curve::new(r_unshifted.knots().clone(), ControlPoints::new(points_b)).unwrap();

    let mut merged = a.clone();
    merged.append(&b).unwrap();

    let mut left_end_constrained = a.clone();
    left_end_constrained.append_constrained(&b, Constraints { left: vec![1.0], right: vec![] }).unwrap();

    let mut right_start_constrained = a.clone();
    right_start_constrained.append_constrained(&b, Constraints { left: vec![], right: vec![0.0] }).unwrap();

    let lim = limits();

    visualization::generate_2d_plot("manipulation/merge-before.svg", vec![(&a, RED_100), (&b, BLUE_100)], &lim, None);
    visualization::generate_2d_plot("manipulation/merge-after.svg", vec![(&merged, PURPLE_100)], &lim, None);
    visualization::generate_2d_plot(
        "manipulation/merge-after-left-end-constrained.svg",
        vec![(&left_end_constrained, PURPLE_100)],
        &lim,
        None,
    );
    visualization::generate_2d_plot(
        "manipulation/merge-after-right-start-constrained.svg",
        vec![(&right_start_constrained, PURPLE_100)],
        &lim,
        None,
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Derivatives
    derivatives_plot();

    // Generation
    manual_plot();
    interpolation_plot();
    fit_plots();

    // Manipulation
    reverse_plots();
    insert_plots();
    split_plots();
    merge_plots();

    Ok(())
}
