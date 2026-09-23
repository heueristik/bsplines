//! Least-squares fitting of data points, with fixed or loose ends and optional penalization.

use nalgebra::{DMatrix, Dyn, SVD};

use crate::{
    Curve,
    error::{Error, Result},
    knots::{KnotMethod, Knots},
    parameters::{ParameterMethod, Parameters},
    points::{ControlPoints, DataPoints},
    svd::decompose,
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

/// Checks the penalization of a fit. The free control points are the control points that the fit places.
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
    debug_assert!(
        knots.polygon_segments() <= points.polyline_segments(),
        "Knots::generate requires at least one data point for each control point"
    );

    if let Some(penalization) = penalization {
        if !(0.0..f64::INFINITY).contains(&penalization.strength) {
            return Err(Error::InvalidPenalizationStrength { strength: penalization.strength });
        }
        if penalization.difference_order >= free_control_points {
            return Err(Error::DifferenceOrderTooLarge {
                difference_order: penalization.difference_order,
                free_control_points,
            });
        }
    }
    Ok(())
}

/// Decomposes the normal matrix of the basis matrix, with the penalty term added. The basis matrix holds one
/// column for each control point that the fit places.
pub(crate) fn decompose_normal_matrix(
    knots: &Knots,
    basis_matrix: &DMatrix<f64>,
    penalization: &Option<Penalization>,
) -> Result<SVD<f64, Dyn, Dyn>> {
    let mut normal_matrix = basis_matrix.transpose() * basis_matrix;

    if let Some(penalization) = penalization {
        let strength = penalization.strength;
        if strength > 0.0 {
            if !knots.is_uniform() {
                return Err(Error::NonUniformKnots);
            }
            let difference_matrix =
                calculate_finite_difference_matrix(penalization.difference_order, basis_matrix.ncols());
            normal_matrix += strength * (difference_matrix.transpose() * difference_matrix);
        }
    }

    decompose(normal_matrix)
}

/// Returns the finite-difference matrix of the order κ for the given number of control points, with one row for
/// each difference of that order — see `Eilers1996`.
fn calculate_finite_difference_matrix(difference_order: usize, control_point_count: usize) -> DMatrix<f64> {
    DMatrix::from_fn(control_point_count - difference_order, control_point_count, |i, j| {
        difference_operator(i, j, difference_order)
    })
}

/// Returns one entry of the finite-difference operator matrix of the order κ — see `Eilers1996`:
///
/// D(κ)ᵢⱼ = (−1)^(κ − j + i) · C(κ, j − i)   for 0 ≤ j − i ≤ κ,   and 0 otherwise
///
/// with the difference order κ and the binomial coefficient C. The order 0 is the identity,
/// and each order takes the first differences of the order below.
fn difference_operator(i: usize, j: usize, difference_order: usize) -> f64 {
    match j.checked_sub(i) {
        Some(offset) if offset <= difference_order => {
            let binomial =
                (0..offset).fold(1.0, |binomial, m| binomial * (difference_order - m) as f64 / (m + 1) as f64);
            if (difference_order - offset).is_multiple_of(2) { binomial } else { -binomial }
        }
        _ => 0.0,
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
    use crate::points::Points;

    use super::*;

    #[test]
    fn finite_difference_matrix_order_0_is_the_identity() {
        let control_point_count = 5;
        let matrix = calculate_finite_difference_matrix(0, control_point_count);
        assert_eq!(matrix, DMatrix::identity(control_point_count, control_point_count));
    }

    #[test]
    fn finite_difference_matrix_order_1() {
        let expected = nalgebra::dmatrix![
            -1.0, 1.0, 0.0, 0.0, 0.0;
             0.0,-1.0, 1.0, 0.0, 0.0;
             0.0, 0.0,-1.0, 1.0, 0.0;
             0.0, 0.0, 0.0,-1.0, 1.0;
        ];
        assert_eq!(calculate_finite_difference_matrix(1, 5), expected);
    }

    #[test]
    fn finite_difference_matrix_order_2() {
        let expected = nalgebra::dmatrix![
             1.0,-2.0, 1.0, 0.0, 0.0;
             0.0, 1.0,-2.0, 1.0, 0.0;
             0.0, 0.0, 1.0,-2.0, 1.0;
        ];
        assert_eq!(calculate_finite_difference_matrix(2, 5), expected);
    }

    #[test]
    fn finite_difference_matrix_rows_of_order_40_sum_to_zero() {
        let difference_order = 40;
        let matrix = calculate_finite_difference_matrix(difference_order, difference_order + 5);

        for row in matrix.row_iter() {
            assert_eq!(row.sum(), 0.0, "the differences of a constant vanish");
        }
    }

    #[test]
    fn build_errors_for_a_penalization_strength_that_is_negative_or_not_finite() {
        let data = test_data_points(8);
        for strength in [-1.0, f64::INFINITY, f64::NAN] {
            let result = Curve::fit(&data, 2).polygon_segments(4).penalized(strength, 2).build();
            assert!(matches!(result, Err(Error::InvalidPenalizationStrength { .. })), "the strength {strength}");
        }
    }

    #[test]
    fn build_errors_when_the_penalty_overflows() {
        let data = test_data_points(8);
        let result = Curve::fit(&data, 2).polygon_segments(4).penalized(f64::MAX, 2).build();
        assert_eq!(result.err(), Some(Error::NonFiniteValue));
    }

    #[test]
    fn fixed_ends_with_one_polygon_segment_connect_the_end_data_points() {
        let data = test_data_points(5);
        let curve = Curve::fit(&data, 1).polygon_segments(1).build().unwrap();

        let last = data.polyline_segments();
        let expected = DMatrix::from_columns(&[data.matrix().column(0), data.matrix().column(last)]);
        assert_eq!(curve.control_points().matrix(), &expected);
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
