//! Least-squares fitting of data points, with fixed or loose ends and optional penalization.

use nalgebra::{Dyn, SVD};

use crate::{
    Curve,
    error::{Error, Result},
    knots::{KnotMethod, Knots},
    parameters::{ParameterMethod, Parameters},
    points::{ControlPoints, DataPoints},
    types::MatD,
};

pub(crate) mod fixed;
pub(crate) mod loose;

/// The penalization of a least-squares fit, see `Eilers1996`.
pub struct Penalization {
    /// The penalization strength λ ≥ 0.
    pub lambda: f64,
    /// The finite-difference order κ of the penalty term.
    pub kappa: usize,
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
    pub fn polygon_segments(mut self, polygon_segments: usize) -> Self {
        self.polygon_segments = Some(polygon_segments);
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
        let polygon_segments = self.polygon_segments.unwrap_or_else(|| self.data.polyline_segments());
        let parameters = Parameters::generate(self.data, ParameterMethod::EquallySpaced);
        let knots = Knots::generate(self.degree, polygon_segments, &parameters, KnotMethod::Uniform)?;

        let points = match self.ends {
            Ends::Fixed => fixed::fit(&knots, self.data, &parameters, self.penalization)?,
            Ends::Loose => loose::fit(&knots, self.data, &parameters, self.penalization)?,
        };
        Curve::new(knots, ControlPoints::new_with_capacity(points, self.degree + 1))
    }
}

fn input_checks(
    knots: &Knots,
    points: &DataPoints,
    parameters: &Parameters,
    penalization: &Option<Penalization>,
) -> Result<()> {
    debug_assert_eq!(
        parameters.polyline_segments(),
        points.polyline_segments(),
        "each data point must have one parameter"
    );

    match (knots.polygon_segments(), points.polyline_segments(), knots.degree(), penalization) {
        (polygon_segments, polyline_segments, _, _) if polygon_segments > polyline_segments => {
            Err(Error::TooFewPolylineSegments { polygon_segments, polyline_segments })
        }
        (polygon_segments, _, degree, _) if polygon_segments < degree => {
            Err(Error::TooFewPolygonSegments { degree, polygon_segments })
        }
        (polygon_segments, _, _, Some(penalization)) if polygon_segments - 1 < penalization.kappa => {
            Err(Error::KappaTooLarge { kappa: penalization.kappa, polygon_segments })
        }
        _ => Ok(()),
    }
}

pub(crate) fn compute_svd(
    knots: &Knots,
    basis_matrix: &MatD,
    penalization: &Option<Penalization>,
    calculate_finite_difference_matrix: Box<dyn FnOnce(usize, &Knots) -> MatD>,
) -> Result<SVD<f64, Dyn, Dyn>> {
    let mut normal_matrix = basis_matrix.transpose() * basis_matrix;

    if let Some(penalization) = penalization {
        let lambda = penalization.lambda;
        if lambda < 0.0 {
            return Err(Error::NegativeLambda { lambda });
        }
        if lambda > 0.0 {
            if !knots.is_uniform() {
                return Err(Error::NonUniformKnots);
            }
            let difference_matrix = calculate_finite_difference_matrix(penalization.kappa, knots);
            normal_matrix += lambda * (difference_matrix.transpose() * difference_matrix);
        }
    }

    Ok(SVD::new(normal_matrix, true, true))
}

/// Returns one entry of the finite-difference operator matrix of order `kappa` — see `Eilers1996`.
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
        kappa if kappa > 1 => difference_operator(i + 1, j, kappa - 1) - difference_operator(i, j, kappa - 1),
        _ => 0,
    }
}

#[cfg(test)]
pub(crate) fn test_data_points(count: usize) -> DataPoints {
    let spacing = 5.0 / count as f64;
    let offset = (count - 1) as f64 * spacing / 2.0;
    DataPoints::new(nalgebra::DMatrix::from_fn(2, count, |row, column| {
        if row == 0 {
            // x coordinates
            column as f64 * spacing - offset
        } else {
            // y coordinates
            if column % 2 == 0 { 0.5 } else { -0.5 }
        }
    }))
}
