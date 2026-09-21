use nalgebra::{Dyn, SVD};

use crate::{
    Curve,
    error::{Error, Result},
    knots,
    knots::{KnotMethod, Knots, is_uniform},
    parameters,
    parameters::{ParameterMethod, Parameters},
    points::{ControlPoints, DataPoints},
    types::MatD,
};

pub(crate) mod fixed;
pub(crate) mod loose;

pub struct Penalization {
    pub lambda: f64,
    pub kappa: usize,
    // TODO add `new` method and assertions according to below
    // A B-spline curve C(u) of degree p can be generated via least-squares minimization and results in
    // an approximation of the (m + 1) data points with dimension N . The number of control points (n + 1)
    // can be specified but must be smaller then the number of data points and greater than the spline degree (m > n ≥
    // p).
}

/// Builds a least-squares fit of data points; created by [`Curve::fit`].
pub struct FitBuilder<'a> {
    data: &'a DataPoints,
    degree: usize,
    polygon_segments: Option<usize>,
    ends: Ends,
    penalization: Option<Penalization>,
}

enum Ends {
    Fixed,
    Loose,
}

impl<'a> FitBuilder<'a> {
    pub(crate) fn new(data: &'a DataPoints, degree: usize) -> Self {
        FitBuilder { data, degree, polygon_segments: None, ends: Ends::Fixed, penalization: None }
    }

    /// Sets the number of polygon segments `n` of the fitted curve.
    /// It defaults to the polyline segments of the data.
    pub fn polygon_segments(mut self, n: usize) -> Self {
        self.polygon_segments = Some(n);
        self
    }

    /// Fixes the curve ends to the first and last data point. This is the default.
    pub fn fixed_ends(mut self) -> Self {
        self.ends = Ends::Fixed;
        self
    }

    /// Lets the curve ends float freely instead of fixing them to the end data points.
    pub fn loose_ends(mut self) -> Self {
        self.ends = Ends::Loose;
        self
    }

    /// Penalizes the fit with the strength `lambda` and the difference order `kappa`, see `Eilers1996`.
    pub fn penalized(mut self, lambda: f64, kappa: usize) -> Self {
        self.penalization = Some(Penalization { lambda, kappa });
        self
    }

    /// Performs the fit.
    pub fn build(self) -> Result<Curve> {
        let n = self.polygon_segments.unwrap_or_else(|| self.data.polyline_segments());
        let params = parameters::generate(self.data, ParameterMethod::EquallySpaced);
        let knots = knots::generate(self.degree, n, &params, KnotMethod::Uniform)?;

        let points = match self.ends {
            Ends::Fixed => fixed::fit(&knots, self.data, &params, self.penalization)?,
            Ends::Loose => loose::fit(&knots, self.data, &params, self.penalization)?,
        };
        Curve::new(knots, ControlPoints::new_with_capacity(points, self.degree + 1))
    }
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

pub(crate) fn compute_svd(
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
