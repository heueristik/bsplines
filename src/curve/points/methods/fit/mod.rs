use nalgebra::{DMatrix, Dyn, SVD};
use thiserror::Error;

use crate::{
    curve::{
        knots::{is_uniform, Knots},
        parameters::Parameters,
        points::DataPoints,
    },
    types::MatD,
};

pub mod fixed;
pub mod loose;

pub enum Method {
    FixedEnds,
    LooseEnds,
}

#[derive(Error, Debug, PartialEq)]
pub enum FitError {
    #[error("The penalization parameter `lambda = {lambda}` cannot be negative.")]
    NegativeLambda { lambda: f64 },

    #[error(
        "The number of data point segments m = {m} must be greater than the requested polynomial segments n = {n}"
    )]
    RequestedPolynomialSegmentAndDataSegementMismatch { n: usize, m: usize },

    #[error(
        "The requested number of polynomial segments n = {n} must be equal or greater than the spline degree p = {p}"
    )]
    RequestedPolynomialSegmentAndSplineDegreeMismatch { n: usize, p: usize },

    #[error("The requested number of polynomial segments n = {n} must be larger than penalization offset kappa kappa = {kappa}")]
    RequestedPolynomialSegmentAndPenalizationKappaMismatch { n: usize, kappa: usize },

    #[error("The number of data point segments m = {m} must be equal to the number of parameter segments mp = {mp}.")]
    DataSegmentsAndParameterSegmentsMismatch { m: usize, mp: usize },

    #[error("Penalization requires equidistant/uniform knots.")] // See `Eilers1996`.
    NonUniformKnots,
}

pub struct Penalization {
    pub lambda: f64,
    pub kappa: usize,
    // TODO add `new` method and assertions according to below
    // A B-spline curve C(u) of degree p can be generated via least-squares minimization and results in
    // an approximation of the (m + 1) data points with dimension N . The number of control points (n + 1)
    // can be specified but must be smaller then the number of data points and greater than the spline degree (m > n ≥
    // p).
}

fn input_checks(
    knots: &Knots,
    points: &DataPoints,
    params: &Parameters,
    penalization: &Option<Penalization>,
) -> Result<(), FitError> {
    match (knots.segments(), points.segments(), params.segments(), knots.degree(), penalization) {
        (n, m, _, _, _) if n > m => Err(FitError::RequestedPolynomialSegmentAndDataSegementMismatch { n, m }),
        (_, m, mp, _, _) if m != mp => Err(FitError::DataSegmentsAndParameterSegmentsMismatch { m, mp }),
        (n, _, _, p, _) if n < p => Err(FitError::RequestedPolynomialSegmentAndSplineDegreeMismatch { n, p }),
        (n, _, _, _, Some(pen)) if n - 1 < pen.kappa => {
            Err(FitError::RequestedPolynomialSegmentAndPenalizationKappaMismatch { n, kappa: pen.kappa })
        }
        _ => Ok(()),
    }
}

pub fn compute_svd(
    knots: &Knots,
    Nmat: &MatD,
    penalization: &Option<Penalization>,
    calculate_finite_difference_matrix: Box<dyn FnOnce(usize, &Knots) -> MatD>,
) -> Result<SVD<f64, Dyn, Dyn>, FitError> {
    let mut mat = Nmat.transpose() * Nmat;

    match penalization {
        Some(penalization) => match penalization.lambda {
            l if l < 0.0 => return Err(FitError::NegativeLambda { lambda: l }),
            l if l > 0.0 && !is_uniform(knots).unwrap() /*TODO refactor errors and eliminate unwrap*/ => return Err(FitError::NonUniformKnots),
            l => {
                let delta_mat = calculate_finite_difference_matrix(penalization.kappa, knots);
                mat += l * (delta_mat.transpose() * delta_mat);
            }
        },
        None => {}
    }

    Ok(SVD::new(mat, true, true))
}

// finite difference matrix penalizing BSplines
fn difference_operator(i: usize, j: usize, kappa: usize) -> isize {
    match kappa {
        1 => {
            if i == j {
                return -1;
            }
            if i + 1 == j {
                return 1;
            }
            0
        }
        k if k > 1 => difference_operator(i + 1, j, k - 1) - difference_operator(i, j, k - 1),
        _ => 0,
    }
}

// TODO remove and use explicit input
fn test_data_points(npoints: usize) -> DataPoints {
    let inc = 5.0 / npoints as f64;
    let shift = (npoints - 1) as f64 * inc / 2.0;
    DataPoints::new(DMatrix::from_fn(2, npoints, |r, c| {
        if r == 0 {
            // x coords
            c as f64 * inc - shift
        } else {
            // y coord
            if c % 2 == 0 {
                0.5
            } else {
                -0.5
            }
        }
    }))
}
