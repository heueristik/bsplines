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
    curve::knots::Knots,
    types::{ControlPointDerivatves, MatD, VecD, VecDView, VecDViewMut},
};

pub mod methods;

#[derive(PartialEq, Debug, Clone)]
pub struct ControlPoints {
    pub(crate) Pk: ControlPointDerivatves,
    k_max: usize,
}

#[derive(PartialEq, Debug, Clone)]
pub struct DataPoints {
    pub(self) matrix: MatD,
}

pub trait Points {
    fn matrix(&self) -> &MatD;
    fn matrix_mut(&mut self) -> &mut MatD;

    fn get(&self, i: usize) -> VecDView {
        self.matrix().column(i)
    }

    fn get_mut(&mut self, i: usize) -> VecDViewMut {
        self.matrix_mut().column_mut(i)
    }

    fn dimension(&self) -> usize {
        self.matrix().nrows()
    }

    fn count(&self) -> usize {
        self.matrix().ncols()
    }

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
    pub fn new(matrix: MatD) -> Self {
        DataPoints { matrix }
    }

    pub fn reverse(&mut self) -> &mut Self {
        reverse(self.matrix_mut());
        self
    }
    pub fn segments(&self) -> usize {
        self.count() - 1
    }
}

impl Points for ControlPoints {
    fn matrix(&self) -> &MatD {
        &self.Pk[0]
    }

    fn matrix_mut(&mut self) -> &mut MatD {
        &mut self.Pk[0]
    }
}

impl ControlPoints {
    pub fn new(points: MatD) -> Self {
        ControlPoints { Pk: vec![points], k_max: 0 }
    }

    pub fn new_with_capacity(points: MatD, capacity: usize) -> ControlPoints {
        let mut Pk: Vec<MatD> = Vec::with_capacity(capacity);
        Pk.push(points);

        ControlPoints { Pk, k_max: 0 }
    }

    pub fn segments(&self) -> usize {
        self.count() - 1
    }

    pub fn matrix_derivative(&self, derivative: usize) -> &MatD {
        assert!(derivative <= self.k_max, "Derivative {} is not calculated", derivative);
        &self.Pk[derivative]
    }

    pub fn matrix_derivative_mut(&mut self, derivative: usize) -> &mut MatD {
        assert!(derivative <= self.k_max, "Derivative {} is not calculated", derivative);
        &mut self.Pk[derivative]
    }

    pub fn count(&self) -> usize {
        self.count_derivative(0)
    }
    pub fn count_derivative(&self, k: usize) -> usize {
        self.Pk[k].ncols()
    }

    pub fn max_derivative(&self) -> usize {
        self.k_max
    }

    pub fn derive(&mut self, knots: &Knots) {
        let p = knots.degree();
        let n = self.segments();

        self.Pk.truncate(1);
        for k in 1..=p {
            let mut Pnew = MatD::zeros(self.dimension(), n - k + 1);
            // TODO iter over points instead
            for (i, mut col) in Pnew.column_iter_mut().enumerate() {
                col.copy_from(&self.derive_single_point(i, k, knots));
            }
            self.Pk.push(Pnew);
        }
        self.k_max = p;
    }

    fn derive_single_point(&self, i: usize, k_max: usize, knots: &Knots) -> VecD {
        let p = knots.degree();

        if k_max == 0 {
            return self.Pk[0].column(i).clone_owned();
        }

        let U0 = knots.vector();
        if U0[i + p + 1] == U0[i + k_max] {
            return VecD::zeros(self.dimension());
        }

        (p - k_max + 1) as f64 / (U0[i + p + 1] - U0[i + k_max])
            * (self.derive_single_point(i + 1, k_max - 1, knots) - self.derive_single_point(i, k_max - 1, knots))
    }

    pub fn reverse(&mut self) -> &mut Self {
        for k in 0..=self.k_max {
            let Pk = self.matrix_derivative_mut(k);
            reverse(Pk);

            // Odd derivatives change their sign upon reversal
            if k % 2 == 1 {
                Pk.mul_assign(-1.0);
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
    fn segments() {
        assert_eq!(control_points_example().segments(), 3);
    }

    #[test]
    fn reverse() {
        assert_eq!(
            control_points_example().reverse().Pk,
            vec![dmatrix![
                7., 5., 3., 1.;
                8., 6., 4., 2.;
            ]]
        );
    }

    /* TODO
    #[test]
    fn reverse_derivatives() {
        todo!()
    }*/
}
