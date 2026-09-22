//! Least-squares fitting of data points, with fixed or loose ends and optional penalization.

use nalgebra::{DMatrix, Dyn, SVD};

use crate::{
    Curve,
    error::{Error, Result},
    knots::{KnotMethod, Knots},
    parameters::{ParameterMethod, Parameters},
    points::{ControlPoints, DataPoints},
};

pub(crate) mod fixed;
pub(crate) mod loose;

/// The penalization of a least-squares fit, see `Eilers1996`.
pub(crate) struct Penalization {
    /// The penalization strength λ ≥ 0.
    pub strength: f64,
    /// The finite-difference order κ of the penalty term.
    pub difference_order: usize,
}

/// Builds a least-squares fit of data points, with fixed or loose ends and an optional penalization;
/// created by [`Curve::fit`].
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

    /// Penalizes the fit with the strength λ and the difference order κ of the penalty term — see `Eilers1996`.
    pub fn penalized(mut self, strength: f64, difference_order: usize) -> Self {
        self.penalization = Some(Penalization { strength, difference_order });
        self
    }

    /// Performs the fit.
    pub fn build(self) -> Result<Curve> {
        let polygon_segments = self.polygon_segments.unwrap_or_else(|| self.data.polyline_segments());
        let parameters = Parameters::generate(self.data, ParameterMethod::EquallySpaced)?;
        let knots = Knots::generate(self.degree, polygon_segments, &parameters, KnotMethod::Uniform)?;

        let points = match self.ends {
            Ends::Fixed => fixed::fit(&knots, self.data, &parameters, self.penalization)?,
            Ends::Loose => loose::fit(&knots, self.data, &parameters, self.penalization)?,
        };
        Curve::new(knots, ControlPoints::new(points))
    }
}

/// Checks the fit input. The free control points are the control points that the fit places.
fn check_input(
    knots: &Knots,
    points: &DataPoints,
    parameters: &Parameters,
    penalization: &Option<Penalization>,
    free_control_points: usize,
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
        (_, _, _, Some(penalization)) if !(0.0..f64::INFINITY).contains(&penalization.strength) => {
            Err(Error::InvalidPenalizationStrength { strength: penalization.strength })
        }
        (_, _, _, Some(penalization)) if penalization.difference_order >= free_control_points => {
            Err(Error::DifferenceOrderTooLarge { difference_order: penalization.difference_order, free_control_points })
        }
        _ => Ok(()),
    }
}

pub(crate) fn decompose_normal_matrix(
    knots: &Knots,
    basis_matrix: &DMatrix<f64>,
    penalization: &Option<Penalization>,
    calculate_finite_difference_matrix: Box<dyn FnOnce(usize, &Knots) -> DMatrix<f64>>,
) -> Result<SVD<f64, Dyn, Dyn>> {
    let mut normal_matrix = basis_matrix.transpose() * basis_matrix;

    if let Some(penalization) = penalization {
        let strength = penalization.strength;
        if strength > 0.0 {
            if !knots.is_uniform() {
                return Err(Error::NonUniformKnots);
            }
            let difference_matrix = calculate_finite_difference_matrix(penalization.difference_order, knots);
            normal_matrix += strength * (difference_matrix.transpose() * difference_matrix);
        }
    }

    Ok(SVD::new(normal_matrix, true, true))
}

/// Returns one entry of the finite-difference operator matrix of the order κ — see `Eilers1996`:
///
/// D(0)ᵢⱼ = δᵢⱼ,   D(κ)ᵢⱼ = D(κ − 1)ᵢ₊₁,ⱼ − D(κ − 1)ᵢⱼ
///
/// with the difference order κ and the Kronecker delta δ.
fn difference_operator(i: usize, j: usize, difference_order: usize) -> isize {
    match difference_order {
        0 => isize::from(i == j),
        _ => difference_operator(i + 1, j, difference_order - 1) - difference_operator(i, j, difference_order - 1),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_errors_for_a_penalization_strength_that_is_negative_or_not_finite() {
        let data = test_data_points(8);
        for strength in [-1.0, f64::INFINITY, f64::NAN] {
            let result = Curve::fit(&data, 2).polygon_segments(4).penalized(strength, 2).build();
            assert!(matches!(result, Err(Error::InvalidPenalizationStrength { .. })), "the strength {strength}");
        }
    }

    #[test]
    fn fixed_ends_allow_difference_orders_below_the_internal_control_points() {
        let data = test_data_points(8);
        let polygon_segments = 3;
        let free_control_points = polygon_segments - 1;
        let fit = |difference_order| {
            Curve::fit(&data, 2).polygon_segments(polygon_segments).penalized(1.0, difference_order).build()
        };

        assert!(fit(free_control_points - 1).is_ok());
        assert_eq!(
            fit(free_control_points).err(),
            Some(Error::DifferenceOrderTooLarge { difference_order: free_control_points, free_control_points })
        );
    }

    #[test]
    fn loose_ends_allow_difference_orders_below_all_control_points() {
        let data = test_data_points(8);
        let polygon_segments = 3;
        let free_control_points = polygon_segments + 1;
        let fit = |difference_order| {
            Curve::fit(&data, 2)
                .polygon_segments(polygon_segments)
                .loose_ends()
                .penalized(1.0, difference_order)
                .build()
        };

        assert!(fit(free_control_points - 1).is_ok());
        assert_eq!(
            fit(free_control_points).err(),
            Some(Error::DifferenceOrderTooLarge { difference_order: free_control_points, free_control_points })
        );
    }
}
