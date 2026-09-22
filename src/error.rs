//! The crate-wide error type.

use thiserror::Error;

/// Convenience alias for a [`core::result::Result`] with the crate-wide [`enum@Error`].
pub type Result<T> = core::result::Result<T, Error>;

/// The error type for all fallible B-spline operations.
///
/// Every operation documents which variants it can produce. The enum is
/// non-exhaustive so that new variants can be added without a breaking release.
#[derive(Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A parameter for curve evaluation lies outside the domain.
    #[error("the parameter u = {u} lies outside the domain [{min}, {max}]")]
    OutsideDomain {
        /// The offending parameter.
        u: f64,
        /// The inclusive lower domain bound.
        min: f64,
        /// The inclusive upper domain bound.
        max: f64,
    },

    /// A parameter for knot insertion or splitting lies outside the open domain interior.
    #[error("the parameter u = {u} lies outside the domain interior ({min}, {max})")]
    OutsideDomainInterior {
        /// The offending parameter.
        u: f64,
        /// The exclusive lower bound.
        min: f64,
        /// The exclusive upper bound.
        max: f64,
    },

    /// The degree must be at least one.
    #[error("the degree p = {degree} must be at least 1")]
    DegreeTooLow {
        /// The offending degree.
        degree: usize,
    },

    /// A curve needs at least p + 1 control points, i.e. at least p polygon segments.
    #[error("the degree p = {degree} exceeds the n = {polygon_segments} polygon segments")]
    TooFewPolygonSegments {
        /// The degree of the curve.
        degree: usize,
        /// The available polygon segments.
        polygon_segments: usize,
    },

    /// The multiplicity of a knot would exceed the degree.
    #[error("the knot u = {u} has multiplicity {multiplicity}, which exceeds the degree p = {degree}")]
    MultiplicityExceedsDegree {
        /// The offending knot value.
        u: f64,
        /// The multiplicity of the knot.
        multiplicity: usize,
        /// The degree of the curve.
        degree: usize,
    },

    /// Two curves to be merged have different degrees.
    #[error("the degrees of the two curves differ: p = {left} vs. p = {right}")]
    DegreeMismatch {
        /// The degree of the left curve.
        left: usize,
        /// The degree of the right curve.
        right: usize,
    },

    /// Two curves to be merged have different dimensions.
    #[error("the dimensions of the two curves differ: {left} vs. {right}")]
    DimensionMismatch {
        /// The dimension of the left curve.
        left: usize,
        /// The dimension of the right curve.
        right: usize,
    },

    /// The knot vector must be clamped.
    #[error("the knot vector must be clamped")]
    UnclampedCurve,

    /// The knot vector must be normalized to the domain [0, 1].
    #[error("the knot vector must be normalized to the domain [0, 1]")]
    UnnormalizedCurve,

    /// Merging supports at most p - 1 constrained points in total.
    #[error(
        "the {total} constrained points must be fewer than the degree p = {degree}, \
         otherwise the linear system has no solution"
    )]
    TooManyConstraints {
        /// The total number of constrained points of both curves.
        total: usize,
        /// The degree of the curves.
        degree: usize,
    },

    /// The penalization strength must not be negative.
    #[error("the penalization parameter lambda = {lambda} must not be negative")]
    NegativeLambda {
        /// The offending penalization strength.
        lambda: f64,
    },

    /// A fit cannot request more polygon segments than the data provides polyline segments.
    #[error(
        "the requested n = {polygon_segments} polygon segments exceed \
         the m = {polyline_segments} polyline segments of the data"
    )]
    TooFewPolylineSegments {
        /// The requested polygon segments of the fitted curve.
        polygon_segments: usize,
        /// The polyline segments of the data.
        polyline_segments: usize,
    },

    /// The penalization difference order must stay below the polygon segments.
    #[error(
        "the penalization difference order kappa = {kappa} must be smaller than the n = {polygon_segments} polygon segments"
    )]
    KappaTooLarge {
        /// The offending difference order.
        kappa: usize,
        /// The requested polygon segments of the fitted curve.
        polygon_segments: usize,
    },

    /// Penalized fitting requires a uniform knot vector (see `Eilers1996`).
    #[error("penalized fitting requires a uniform knot vector")]
    NonUniformKnots,
}
