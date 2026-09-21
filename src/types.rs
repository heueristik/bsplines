//! Numeric type aliases over [nalgebra] with dynamic dimensions and `f64` scalars.

use nalgebra::{Dyn, MatrixView, MatrixViewMut, OMatrix, OVector, U1};

/// A dynamically sized column vector.
pub type VecD = OVector<f64, Dyn>;

/// An immutable view into a [`VecD`].
pub type VecDView<'a> = MatrixView<'a, f64, Dyn, U1, U1, Dyn>;
/// A mutable view into a [`VecD`].
pub type VecDViewMut<'a> = MatrixViewMut<'a, f64, Dyn, U1, U1, Dyn>;

/// A dynamically sized matrix.
pub type MatD = OMatrix<f64, Dyn, Dyn>;

/// Eigen-style sub-vector accessors for [`VecD`].
pub trait VecHelpers {
    /// Returns a view of the first `count` elements.
    fn head(&self, count: usize) -> MatrixView<'_, f64, Dyn, U1, U1, Dyn>;
    /// Returns a mutable view of the first `count` elements.
    fn head_mut(&mut self, count: usize) -> MatrixViewMut<'_, f64, Dyn, U1, U1, Dyn>;

    /// Returns a view of `count` elements starting at index `start`.
    fn segment(&self, start: usize, count: usize) -> MatrixView<'_, f64, Dyn, U1, U1, Dyn>;
    /// Returns a mutable view of `count` elements starting at index `start`.
    fn segment_mut(&mut self, start: usize, count: usize) -> MatrixViewMut<'_, f64, Dyn, U1, U1, Dyn>;

    /// Returns a view of the last `count` elements.
    fn tail(&self, count: usize) -> MatrixView<'_, f64, Dyn, U1, U1, Dyn>;
    /// Returns a mutable view of the last `count` elements.
    fn tail_mut(&mut self, count: usize) -> MatrixViewMut<'_, f64, Dyn, U1, U1, Dyn>;
}

impl VecHelpers for VecD {
    fn head(&self, count: usize) -> MatrixView<'_, f64, Dyn, U1, U1, Dyn> {
        self.segment(0, count)
    }

    fn head_mut(&mut self, count: usize) -> MatrixViewMut<'_, f64, Dyn, U1, U1, Dyn> {
        self.segment_mut(0, count)
    }

    fn segment(&self, start: usize, count: usize) -> MatrixView<'_, f64, Dyn, U1, U1, Dyn> {
        self.generic_view((start, 0), (Dyn(count), U1))
    }

    fn segment_mut(&mut self, start: usize, count: usize) -> MatrixViewMut<'_, f64, Dyn, U1, U1, Dyn> {
        self.generic_view_mut((start, 0), (Dyn(count), U1))
    }

    fn tail(&self, count: usize) -> MatrixView<'_, f64, Dyn, U1, U1, Dyn> {
        self.segment(self.len() - count, count)
    }

    fn tail_mut(&mut self, count: usize) -> MatrixViewMut<'_, f64, Dyn, U1, U1, Dyn> {
        self.segment_mut(self.len() - count, count)
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
