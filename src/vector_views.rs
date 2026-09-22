//! Views of the first, middle, and last elements of a column vector.

use nalgebra::{DVector, DVectorView, DVectorViewMut, Dyn, U1};

/// Views of the first, middle, and last elements of a column vector.
pub(crate) trait VectorViews {
    /// Returns a view of the first `count` elements.
    fn head(&self, count: usize) -> DVectorView<'_, f64>;
    /// Returns a mutable view of the first `count` elements.
    fn head_mut(&mut self, count: usize) -> DVectorViewMut<'_, f64>;

    /// Returns a view of `count` elements starting at index `start`.
    fn segment(&self, start: usize, count: usize) -> DVectorView<'_, f64>;
    /// Returns a mutable view of `count` elements starting at index `start`.
    fn segment_mut(&mut self, start: usize, count: usize) -> DVectorViewMut<'_, f64>;

    /// Returns a view of the last `count` elements.
    fn tail(&self, count: usize) -> DVectorView<'_, f64>;
    /// Returns a mutable view of the last `count` elements.
    fn tail_mut(&mut self, count: usize) -> DVectorViewMut<'_, f64>;
}

impl VectorViews for DVector<f64> {
    fn head(&self, count: usize) -> DVectorView<'_, f64> {
        self.segment(0, count)
    }

    fn head_mut(&mut self, count: usize) -> DVectorViewMut<'_, f64> {
        self.segment_mut(0, count)
    }

    fn segment(&self, start: usize, count: usize) -> DVectorView<'_, f64> {
        self.generic_view((start, 0), (Dyn(count), U1))
    }

    fn segment_mut(&mut self, start: usize, count: usize) -> DVectorViewMut<'_, f64> {
        self.generic_view_mut((start, 0), (Dyn(count), U1))
    }

    fn tail(&self, count: usize) -> DVectorView<'_, f64> {
        self.segment(self.len() - count, count)
    }

    fn tail_mut(&mut self, count: usize) -> DVectorViewMut<'_, f64> {
        self.segment_mut(self.len() - count, count)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dvector;

    use super::*;

    fn example() -> DVector<f64> {
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
