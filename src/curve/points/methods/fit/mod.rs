use nalgebra::{Dyn, SVD};

use crate::{
    curve::{
        knots::{Knots, is_uniform},
        parameters::Parameters,
        points::DataPoints,
    },
    error::{Error, Result},
    types::MatD,
};

pub mod fixed;
pub mod loose;

pub enum Method {
    FixedEnds,
    LooseEnds,
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
) -> Result<()> {
    match (
        knots.polygon_segments(),
        points.polyline_segments(),
        params.polyline_segments(),
        knots.degree(),
        penalization,
    ) {
        (n, m, _, _, _) if n > m => Err(Error::TooFewPolylineSegments { polygon_segments: n, polyline_segments: m }),
        (_, m, mp, _, _) if m != mp => {
            Err(Error::ParameterSegmentsMismatch { polyline_segments: m, parameter_segments: mp })
        }
        (n, _, _, p, _) if n < p => Err(Error::TooFewPolygonSegments { degree: p, polygon_segments: n }),
        (n, _, _, _, Some(pen)) if n - 1 < pen.kappa => {
            Err(Error::KappaTooLarge { kappa: pen.kappa, polygon_segments: n })
        }
        _ => Ok(()),
    }
}

pub fn compute_svd(
    knots: &Knots,
    n_mat: &MatD,
    penalization: &Option<Penalization>,
    calculate_finite_difference_matrix: Box<dyn FnOnce(usize, &Knots) -> MatD>,
) -> Result<SVD<f64, Dyn, Dyn>> {
    let mut mat = n_mat.transpose() * n_mat;

    if let Some(penalization) = penalization {
        let lambda = penalization.lambda;
        if lambda < 0.0 {
            return Err(Error::NegativeLambda { lambda });
        }
        if lambda > 0.0 {
            if !is_uniform(knots)? {
                return Err(Error::NonUniformKnots);
            }
            let delta_mat = calculate_finite_difference_matrix(penalization.kappa, knots);
            mat += lambda * (delta_mat.transpose() * delta_mat);
        }
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
#[cfg(test)]
pub(crate) fn test_data_points(npoints: usize) -> DataPoints {
    let inc = 5.0 / npoints as f64;
    let shift = (npoints - 1) as f64 * inc / 2.0;
    DataPoints::new(nalgebra::DMatrix::from_fn(2, npoints, |r, c| {
        if r == 0 {
            // x coords
            c as f64 * inc - shift
        } else {
            // y coord
            if c % 2 == 0 { 0.5 } else { -0.5 }
        }
    }))
}
