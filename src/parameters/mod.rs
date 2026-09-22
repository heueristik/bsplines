//! Implements different parameter generation methods.
//!
//! - Equally spaced parameters
//! - Centripetal method
//! - Chord-length method

use nalgebra::DVector;

use crate::{
    error::{Error, Result},
    points::{DataPoints, Points},
};

pub(crate) mod methods;

/// The parameter values ū assigned to the data points for interpolation and fitting,
/// one per data point.
#[derive(Debug, Clone)]
pub struct Parameters {
    vector: DVector<f64>,
}

impl Parameters {
    /// Generates the parameters ū for the data points with the given method.
    /// The data needs at least two points with finite coordinates.
    pub fn generate(data: &DataPoints, method: ParameterMethod) -> Result<Self> {
        let count = data.count();
        if count < 2 {
            return Err(Error::TooFewDataPoints { count });
        }
        if let Some(index) = data.matrix().column_iter().position(|point| point.iter().any(|x| !x.is_finite())) {
            return Err(Error::NonFiniteDataPoint { index });
        }

        match method {
            ParameterMethod::EquallySpaced => Ok(methods::equally_spaced(data.polyline_segments())),
            ParameterMethod::ChordLength => methods::chord_length(data),
            ParameterMethod::Centripetal => methods::centripetal(data),
        }
    }

    /// Returns parameters from the given values, one per data point.
    pub fn new(vector: DVector<f64>) -> Self {
        Parameters { vector }
    }

    /// Returns the parameter values ū.
    pub fn vector(&self) -> &DVector<f64> {
        &self.vector
    }

    /// Returns the number of polyline segments m of the data the parameters belong to —
    /// one less than the number of parameters.
    pub fn polyline_segments(&self) -> usize {
        self.vector.len() - 1
    }
}

/// The method assigning a parameter value ū to every data point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParameterMethod {
    /// Distributes the parameters equally — eq. (9.3) in `Piegl1997`.
    /// Simple, but risks erratic shapes when the data is unevenly spaced.
    EquallySpaced,
    /// Distributes the parameters by the square roots of the chord lengths — eq. (9.6) in `Piegl1997`.
    /// Dampens the effect of outlier points on the parametrization.
    Centripetal,
    /// Distributes the parameters proportionally to the chord lengths — eq. (9.5) in `Piegl1997`.
    /// The most common choice, approximating a uniform parametrization with respect to arc length.
    ChordLength,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, dmatrix, dvector};

    use super::*;

    #[test]
    fn generate_errors_for_one_data_point() {
        let data = DataPoints::new(dmatrix![1.0; 2.0]);
        assert_eq!(
            Parameters::generate(&data, ParameterMethod::EquallySpaced).err(),
            Some(Error::TooFewDataPoints { count: 1 })
        );
    }

    #[test]
    fn generate_errors_for_a_coordinate_that_is_not_finite() {
        let mut matrix = DMatrix::from_fn(2, 4, |row, column| (row + column) as f64);
        let index = 2;
        matrix[(1, index)] = f64::NAN;
        assert_eq!(
            Parameters::generate(&DataPoints::new(matrix), ParameterMethod::EquallySpaced).err(),
            Some(Error::NonFiniteDataPoint { index })
        );
    }

    #[test]
    fn chord_length_and_centripetal_error_for_coincident_data_points() {
        let data = DataPoints::new(DMatrix::from_element(2, 4, 1.0));
        assert!(
            Parameters::generate(&data, ParameterMethod::EquallySpaced).is_ok(),
            "equally spaced parameters do not depend on the positions"
        );

        for method in [ParameterMethod::ChordLength, ParameterMethod::Centripetal] {
            assert_eq!(Parameters::generate(&data, method).err(), Some(Error::CoincidentDataPoints));
        }
    }

    #[test]
    fn centripetal_accepts_closely_spaced_data_points() {
        let spacing = 1e-18;
        let data = DataPoints::new(DMatrix::from_fn(1, 4, |_, column| column as f64 * spacing));
        let parameters = Parameters::generate(&data, ParameterMethod::Centripetal).unwrap();
        assert_relative_eq!(
            parameters.vector(),
            &dvector![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            epsilon = f64::EPSILON.sqrt()
        );
    }

    #[test]
    fn chord_length_errors_when_the_chord_lengths_overflow() {
        let data = DataPoints::new(dmatrix![0.0, 1e200, 2e200;]);
        assert_eq!(Parameters::generate(&data, ParameterMethod::ChordLength).err(), Some(Error::NonFiniteValue));
    }
}
