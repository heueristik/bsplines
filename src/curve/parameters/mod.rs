//! Implements different parameter generation methods.
//!
//! - Equally spaced parameters
//! - Centripetal method
//! - Chord-length method

use crate::{curve::points::DataPoints, types::VecD};

pub mod methods;

#[derive(Debug, Clone)]
pub struct Parameters {
    vector: VecD,
    polyline_segments: usize,
}

impl Parameters {
    pub fn new(vector: VecD, polyline_segments: usize) -> Self {
        Parameters { vector, polyline_segments }
    }

    pub fn vector(&self) -> &VecD {
        &self.vector
    }

    pub fn polyline_segments(&self) -> usize {
        self.polyline_segments
    }
}

pub enum Method {
    EquallySpaced,
    Centripetal,
    ChordLength,
}

pub fn generate(points: &DataPoints, method: Method) -> Parameters {
    match method {
        Method::EquallySpaced => methods::equally_spaced(points.polyline_segments()),
        Method::ChordLength => methods::chord_length(points),
        Method::Centripetal => methods::centripetal(points),
    }
}
