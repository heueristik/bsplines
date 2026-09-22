//! Implements different parameter generation methods.
//!
//! - Equally spaced parameters
//! - Centripetal method
//! - Chord-length method

use crate::{points::DataPoints, types::VecD};

pub(crate) mod methods;

/// The parameter values ū assigned to the data points for interpolation and fitting,
/// one per data point.
#[derive(Debug, Clone)]
pub struct Parameters {
    vector: VecD,
    polyline_segments: usize,
}

impl Parameters {
    /// Returns parameters from the given values, associated with a data polyline of `m` segments.
    pub fn new(vector: VecD, polyline_segments: usize) -> Self {
        Parameters { vector, polyline_segments }
    }

    /// Returns the parameter values ū.
    pub fn vector(&self) -> &VecD {
        &self.vector
    }

    /// Returns the number of polyline segments `m` of the data the parameters belong to.
    pub fn polyline_segments(&self) -> usize {
        self.polyline_segments
    }
}

/// The method assigning a parameter value ū to every data point.
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

/// Generates the parameters ū for the data points with the given method.
pub fn generate(points: &DataPoints, method: ParameterMethod) -> Parameters {
    match method {
        ParameterMethod::EquallySpaced => methods::equally_spaced(points.polyline_segments()),
        ParameterMethod::ChordLength => methods::chord_length(points),
        ParameterMethod::Centripetal => methods::centripetal(points),
    }
}
