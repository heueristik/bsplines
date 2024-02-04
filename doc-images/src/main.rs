use std::ops::AddAssign;

use nalgebra::{dmatrix, dvector};
use plotters::{prelude::*, style::full_palette::TEAL};

use bsplines::{
    curve::{
        generation::{
            generate,
            Generation::{Interpolation, LeastSquaresFit, Manual},
        },
        knots::Generation::Uniform,
        points::{
            methods::fit::{Method::LooseEnds, Penalization},
            ControlPoints, DataPoints, Points,
        },
        Curve,
    },
    manipulation::{
        merge::{merge, merge_from, merge_to},
        split::split_and_normalize,
    },
};

use crate::visualization::Limits;

mod visualization;

const PLOTS_DIR: &str = "doc-images/plots/";

const RED_50: RGBAColor = RGBAColor(255, 0, 0, 0.5);
const RED_100: RGBAColor = RGBAColor(255, 0, 0, 1.0);
const BLUE_50: RGBAColor = RGBAColor(0, 0, 255, 0.5);
const BLUE_100: RGBAColor = RGBAColor(0, 0, 255, 1.0);

const PURPLE_50: RGBAColor = RGBAColor(200, 0, 200, 0.5);
const PURPLE_100: RGBAColor = RGBAColor(200, 0, 200, 1.0);

fn limits() -> Limits {
    Limits { min: vec![-3.0, -3.0], max: vec![3.0, 3.0] }
}

fn scatteredDataPoints() -> DataPoints {
    let dp = DataPoints::new(dmatrix![
        -2.50,-2.45,-2.15,-1.70,-1.50,-1.35,-1.20, 0.05, 0.20, 0.55, 0.65, 1.00, 1.20, 1.50, 1.75, 2.00, 2.15, 2.50;
        -2.55,-2.10,-2.45,-2.60,-2.15,-2.15,-1.85,-1.20,-0.70,-0.90,-0.20, 2.00, 0.95, 1.40,-0.70,-1.90,-1.70,-2.15;
    ]);

    dp
}

fn example_spline(p: usize) -> Curve {
    generate(Manual {
        degree: p,
        points: ControlPoints::new(dmatrix![
            -2.5,-1.5,-0.5, 1.0, 2.0, 0.0;
            -2.5, 1.0,-1.5,-2.0, 1.0, 2.0;
        ]),
        knots: Uniform,
    })
    .unwrap()
}

fn fit_plots() {
    let p = 2;
    let dp = scatteredDataPoints();
    let bs_max = generate(LeastSquaresFit {
        degree: p,
        points: &dp,
        intended_segments: dp.segments(),
        method: LooseEnds,
        penalization: None,
    })
    .unwrap();

    let bs_half = generate(LeastSquaresFit {
        degree: p,
        points: &dp,
        intended_segments: dp.count() / 3,
        method: LooseEnds,
        penalization: None,
    })
    .unwrap();

    let bs_half_penalized = generate(LeastSquaresFit {
        degree: p,
        points: &dp,
        intended_segments: dp.count() / 3,
        method: LooseEnds,
        penalization: Some(Penalization { lambda: 0.5, kappa: 2 }),
    })
    .unwrap();

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
    let dp = scatteredDataPoints();
    let c = generate(Interpolation { degree: 2, points: &dp }).unwrap();

    visualization::generate_2d_plot("generation/interpolation.svg", vec![(&c, RED_100)], &limits(), Some(&dp));
}

fn manual_plot() {
    let dp = scatteredDataPoints();
    let c = generate(Manual { degree: 2, knots: Uniform, points: ControlPoints::new(dp.matrix().clone()) }).unwrap();
    visualization::generate_2d_plot("generation/manual.svg", vec![(&c, RED_100)], &limits(), Some(&dp));
}

fn derivatives_plot() {
    let mut bs_k0 = generate(Manual {
        degree: 3,
        points: ControlPoints::new(dmatrix![
            -0.25, -0.05, 0.0, 0.05, 0.25;
        ]),
        knots: Uniform,
    })
    .unwrap();

    let lim = Limits { min: vec![-4.4], max: vec![9.0] };

    let bs_k1 = bs_k0.get_derivative_curve(1);
    let bs_k2 = bs_k0.get_derivative_curve(2);
    let bs_k3 = bs_k0.get_derivative_curve(3);

    visualization::generate_1d_plot(
        "derivatives.svg",
        vec![(&bs_k0, RED_100), (&bs_k1, PURPLE_100), (&bs_k2, BLUE_100), (&bs_k3, TEAL.to_rgba())],
        &lim,
    );

    bs_k0.reverse(); // p = 3 CORRECT
    let bs_k1_rev = bs_k0.get_derivative_curve(1); // p = 2 WRONG
    let bs_k2_rev = bs_k0.get_derivative_curve(2); // p = 1 CORRECT
    let bs_k3_rev = bs_k0.get_derivative_curve(3); // p = 0 WRONG

    visualization::generate_1d_plot(
        "derivatives-reversed.svg",
        vec![(&bs_k0, RED_100), (&bs_k1_rev, PURPLE_100), (&bs_k2_rev, BLUE_100), (&bs_k3_rev, TEAL.to_rgba())],
        &lim,
    );
}

fn insert_plots() {
    let p = 2;
    let mut c = example_spline(p);
    let lim = Limits { min: vec![-3., -3.], max: vec![3., 3.] };
    let u = 0.8;
    visualization::generate_2d_plot("manipulation/insert-before.svg", vec![(&c, RED_100)], &lim, None);

    c.insert_times(u, 1).unwrap();

    visualization::generate_2d_plot("manipulation/insert-after.svg", vec![(&c, BLUE_100)], &lim, None);
}

fn split_plots() {
    let c = example_spline(2);
    let lim = Limits { min: vec![-3., -3.], max: vec![3., 3.] };
    visualization::generate_2d_plot("manipulation/split-before.svg", vec![(&c, PURPLE_100)], &lim, None);
    let (a, b) = split_and_normalize(&c, 0.5, (true, true)).unwrap();
    visualization::generate_2d_plot("manipulation/split-after.svg", vec![(&a, RED_100), (&b, BLUE_100)], &lim, None);
}

fn reverse_plots() {
    let mut c = example_spline(2);
    let lim = Limits { min: vec![-3., -3.], max: vec![3., 3.] };
    visualization::generate_2d_plot("manipulation/reverse-before.svg", vec![(&c, RED_100)], &lim, None);
    c.reverse();
    visualization::generate_2d_plot("manipulation/reverse-after.svg", vec![(&c, PURPLE_100)], &lim, None);
}

fn merge_plots() {
    let c = example_spline(2);
    let (l_unshifted, r_unshifted) = split_and_normalize(&c, 0.5, (true, true)).unwrap();

    let mut P_a = l_unshifted.points.matrix().clone();
    let mut P_b = r_unshifted.points.matrix().clone();

    // Shift points
    P_a.column_mut(P_a.ncols() - 1).add_assign(dvector![0.25, -0.25]);
    P_b.column_mut(0).add_assign(dvector![0.25, 0.25]);

    let a = Curve::new(l_unshifted.knots.clone(), ControlPoints::new(P_a)).unwrap();
    let b = Curve::new(r_unshifted.knots.clone(), ControlPoints::new(P_b)).unwrap();

    let lim = limits();

    visualization::generate_2d_plot("manipulation/merge-before.svg", vec![(&a, RED_100), (&b, BLUE_100)], &lim, None);
    visualization::generate_2d_plot(
        "manipulation/merge-after.svg",
        vec![(&merge(&a, &b).unwrap(), PURPLE_100)],
        &lim,
        None,
    );
    visualization::generate_2d_plot(
        "manipulation/merge-after-left-end-constrained.svg",
        vec![(&merge_from(&a, &b).unwrap(), PURPLE_100)],
        &lim,
        None,
    );
    visualization::generate_2d_plot(
        "manipulation/merge-after-right-start-constrained.svg",
        vec![(&merge_to(&a, &b).unwrap(), PURPLE_100)],
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
