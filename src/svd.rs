//! Decomposes the linear systems of the interpolation, the fits, and the merge.

use nalgebra::{DMatrix, Dyn, SVD};

use crate::error::{Error, Result};

/// Returns the singular value decomposition of the matrix. A matrix with a value that is not finite
/// gives an error: nalgebra's decomposition of it panics or never returns.
pub(crate) fn decompose(matrix: DMatrix<f64>) -> Result<SVD<f64, Dyn, Dyn>> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteValue);
    }
    Ok(SVD::new(matrix, true, true))
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::*;

    #[test]
    fn decompose_errors_for_a_value_that_is_not_finite() {
        assert_eq!(decompose(dmatrix![f64::NAN, 1.0; 1.0, 1.0]).err(), Some(Error::NonFiniteValue));
        assert_eq!(decompose(dmatrix![f64::INFINITY, 1.0; 1.0, 1.0]).err(), Some(Error::NonFiniteValue));
    }
}
