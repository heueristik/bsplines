use nalgebra::{Dyn, Matrix, MatrixView, MatrixViewMut, OMatrix, OVector, Owned, U1};

pub type VecD = OVector<f64, Dyn>;
pub type RowVecD = Matrix<f64, U1, Dyn, Owned<f64, U1, Dyn>>;

pub type VecDView<'a> = MatrixView<'a, f64, Dyn, U1, U1, Dyn>;
pub type VecDViewMut<'a> = MatrixViewMut<'a, f64, Dyn, U1, U1, Dyn>;

pub type MatD = OMatrix<f64, Dyn, Dyn>;
pub type MatDView<'a> = MatrixView<'a, f64, Dyn, Dyn>;
pub type MatRowD = OMatrix<f64, U1, Dyn>;

pub type KnotVectorDerivatives = Vec<VecD>;
pub type ControlPointDerivatves = Vec<MatD>;

pub trait VecHelpers {
    fn head(&self, n: usize) -> MatrixView<f64, Dyn, U1, U1, Dyn>;
    fn head_mut(&mut self, n: usize) -> MatrixViewMut<f64, Dyn, U1, U1, Dyn>;

    fn segment(&self, i: usize, n: usize) -> MatrixView<f64, Dyn, U1, U1, Dyn>;
    fn segment_mut(&mut self, i: usize, n: usize) -> MatrixViewMut<f64, Dyn, U1, U1, Dyn>;

    fn tail(&self, n: usize) -> MatrixView<f64, Dyn, U1, U1, Dyn>;
    fn tail_mut(&mut self, n: usize) -> MatrixViewMut<f64, Dyn, U1, U1, Dyn>;
}

impl VecHelpers for VecD {
    fn head(&self, n: usize) -> MatrixView<f64, Dyn, U1, U1, Dyn> {
        self.segment(0, n)
    }

    fn head_mut(&mut self, n: usize) -> MatrixViewMut<f64, Dyn, U1, U1, Dyn> {
        self.segment_mut(0, n)
    }

    fn segment(&self, start: usize, n: usize) -> MatrixView<f64, Dyn, U1, U1, Dyn> {
        self.generic_view((start, 0), (Dyn(n), U1))
    }

    fn segment_mut(&mut self, start: usize, n: usize) -> MatrixViewMut<f64, Dyn, U1, U1, Dyn> {
        self.generic_view_mut((start, 0), (Dyn(n), U1))
    }

    fn tail(&self, n: usize) -> MatrixView<f64, Dyn, U1, U1, Dyn> {
        self.segment(self.len() - n, n)
    }

    fn tail_mut(&mut self, n: usize) -> MatrixViewMut<f64, Dyn, U1, U1, Dyn> {
        self.segment_mut(self.len() - n, n)
    }
}

#[cfg(test)]
mod vec_helpers {
    use nalgebra::dvector;

    use super::*;

    fn example() -> VecD {
        dvector![0.0, 1.0, 2.0, 3.0]
    }

    #[test]
    fn head() {
        assert_eq!(example().head(2).as_slice(), [0.0, 1.0]);
    }

    #[test]
    fn segment() {
        assert_eq!(example().segment(1, 2).as_slice(), [1.0, 2.0]);
    }

    #[test]
    fn tail() {
        assert_eq!(example().tail(2).as_slice(), [2.0, 3.0]);
    }
}
