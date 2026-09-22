#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("eq-control-points", "doc-images/equations/control-points.svg")))]
//! Implements the control points constituting the control polygon of the curve.
//!
//! The control points of the `k`-th derivative B-spline curve can be derived
//! from the zeroth order control points
//!
//! ![The control points][eq-control-points]
//!
//! Control points are generated and manipulated as part of different curve generation
//! and curve manipulation methods.

use std::ops::MulAssign;

use crate::{
    knots::Knots,
    types::{MatD, VecD, VecDView, VecDViewMut},
};

/// The control points P of a curve and of its derivatives; together they form the control polygon.
#[derive(PartialEq, Debug, Clone)]
pub struct ControlPoints {
    /// The control point matrices of the curve and of its derivatives, indexed by derivative order.
    pub(crate) derivatives: Vec<MatD>,
}

/// Input points that a curve is interpolated through or fitted to.
/// Data points are consumed by curve generation; they are not part of the resulting curve.
#[derive(PartialEq, Debug, Clone)]
pub struct DataPoints {
    matrix: MatD,
}

/// Common accessors for point sets stored as one column per point.
pub trait Points {
    /// Returns the coordinate matrix holding one point per column.
    fn matrix(&self) -> &MatD;
    /// Returns the mutable coordinate matrix holding one point per column.
    fn matrix_mut(&mut self) -> &mut MatD;

    /// Returns a view of the `i`-th point.
    fn get(&self, index: usize) -> VecDView<'_> {
        self.matrix().column(index)
    }

    /// Returns a mutable view of the `i`-th point.
    fn get_mut(&mut self, index: usize) -> VecDViewMut<'_> {
        self.matrix_mut().column_mut(index)
    }

    /// Returns the dimension N of the points.
    fn dimension(&self) -> usize {
        self.matrix().nrows()
    }

    /// Returns the number of points.
    fn count(&self) -> usize {
        self.matrix().ncols()
    }

    /// Returns whether there are no points.
    fn is_empty(&self) -> bool {
        self.matrix().is_empty()
    }
}

impl Points for DataPoints {
    fn matrix(&self) -> &MatD {
        &self.matrix
    }

    fn matrix_mut(&mut self) -> &mut MatD {
        &mut self.matrix
    }
}

impl DataPoints {
    /// Returns data points from a coordinate matrix holding one point per column.
    pub fn new(matrix: MatD) -> Self {
        DataPoints { matrix }
    }

    /// Reverses the order of the points.
    pub fn reverse(&mut self) -> &mut Self {
        reverse(self.matrix_mut());
        self
    }

    /// Returns the number of chords m of the data polyline — one less than the number of points.
    pub fn polyline_segments(&self) -> usize {
        self.count() - 1
    }
}

impl Points for ControlPoints {
    fn matrix(&self) -> &MatD {
        &self.derivatives[0]
    }

    fn matrix_mut(&mut self) -> &mut MatD {
        &mut self.derivatives[0]
    }
}

impl ControlPoints {
    /// Returns control points from a coordinate matrix holding one point per column.
    pub fn new(points: MatD) -> Self {
        ControlPoints { derivatives: vec![points] }
    }

    /// Returns the number of segments n of the control polygon — one less than the number of points.
    pub fn polygon_segments(&self) -> usize {
        self.count() - 1
    }

    /// Returns the control point matrix of the `k`-th derivative curve.
    pub(crate) fn matrix_derivative(&self, derivative: usize) -> &MatD {
        &self.derivatives[derivative]
    }

    /// Derives the control points of all derivative orders from the curve's control points —
    /// see the formula in the [module documentation][self].
    pub(crate) fn derive(&mut self, knots: &Knots) {
        let degree = knots.degree();
        let polygon_segments = self.polygon_segments();

        self.derivatives.truncate(1);
        for derivative in 1..=degree {
            let mut new_points = MatD::zeros(self.dimension(), polygon_segments - derivative + 1);
            for (i, mut column) in new_points.column_iter_mut().enumerate() {
                column.copy_from(&self.derive_single_point(i, derivative, knots));
            }
            self.derivatives.push(new_points);
        }
    }

    fn derive_single_point(&self, index: usize, derivative: usize, knots: &Knots) -> VecD {
        let degree = knots.degree();

        if derivative == 0 {
            return self.derivatives[0].column(index).clone_owned();
        }

        let knot_values = knots.vector();
        if knot_values[index + degree + 1] == knot_values[index + derivative] {
            return VecD::zeros(self.dimension());
        }

        (degree - derivative + 1) as f64 / (knot_values[index + degree + 1] - knot_values[index + derivative]) *
            (self.derive_single_point(index + 1, derivative - 1, knots) -
                self.derive_single_point(index, derivative - 1, knots))
    }

    /// Reverses the order of the points of all derivative orders.
    /// The odd derivative matrices also change their sign.
    pub(crate) fn reverse(&mut self) -> &mut Self {
        for (derivative, matrix) in self.derivatives.iter_mut().enumerate() {
            reverse(matrix);

            if derivative % 2 == 1 {
                matrix.mul_assign(-1.0);
            }
        }
        self
    }
}

pub(crate) fn reverse(points: &mut MatD) {
    let ncols = points.ncols();
    let half_ncols = points.ncols() / 2;

    for i in 0..half_ncols {
        points.swap_columns(i, ncols - 1 - i);
    }
}

pub(crate) fn reversed(points: &MatD) -> MatD {
    let mut copy = points.clone();
    reverse(&mut copy);
    copy
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::*;

    fn control_points_example() -> ControlPoints {
        ControlPoints::new(dmatrix![
            1., 3., 5., 7.;
            2., 4., 6., 8.;
        ])
    }

    #[test]
    fn dimension() {
        assert_eq!(control_points_example().dimension(), 2);
    }

    #[test]
    fn count() {
        assert_eq!(control_points_example().count(), 4);
    }

    #[test]
    fn polygon_segments() {
        assert_eq!(control_points_example().polygon_segments(), 3);
    }

    #[test]
    fn reverse() {
        assert_eq!(
            control_points_example().reverse().derivatives,
            vec![dmatrix![
                7., 5., 3., 1.;
                8., 6., 4., 2.;
            ]]
        );
    }
}
