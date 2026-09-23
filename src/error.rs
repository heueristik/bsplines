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

    /// A knot vector of degree p needs at least 2p + 2 knots: the smallest curve of degree p has p + 1 control points.
    #[error("the knot vector holds {count} knots, but the degree p = {degree} needs at least 2p + 2 knots")]
    TooFewKnots {
        /// The number of knots.
        count: usize,
        /// The degree of the curve.
        degree: usize,
    },

    /// A knot is NaN or infinite.
    #[error("the knot at index {index} is not finite")]
    NonFiniteKnot {
        /// The index of the knot.
        index: usize,
    },

    /// Knots must be in non-decreasing order.
    #[error("the knot at index {index} is smaller than the knot before it")]
    DecreasingKnots {
        /// The index of the smaller knot.
        index: usize,
    },

    /// The domain of a knot vector runs from knot p to knot n + 1 and needs a positive length.
    #[error("the knots p and n + 1 are equal, so the domain of the knot vector has length zero")]
    ZeroLengthDomain,

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

    /// The derivative order exceeds the degree: the k-th derivative curve would have the negative degree p − k.
    #[error("the derivative order k = {derivative} exceeds the degree p = {degree}")]
    DerivativeExceedsDegree {
        /// The requested derivative order.
        derivative: usize,
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

    /// A curve with n + p + 2 knots needs n + 1 control points: one for each basis function.
    #[error("the knot vector needs {expected} control points, but there are {count}")]
    ControlPointCountMismatch {
        /// The number of basis functions of the knot vector.
        expected: usize,
        /// The number of control points.
        count: usize,
    },

    /// A control point has a coordinate that is NaN or infinite.
    #[error("the control point at index {index} has a coordinate that is not finite")]
    NonFiniteControlPoint {
        /// The index of the control point.
        index: usize,
    },

    /// The knot vector must be clamped: exactly its first p + 1 knots are equal, and exactly its last p + 1 knots.
    #[error("the knot vector must be clamped: exactly its first p + 1 and its last p + 1 knots are equal")]
    UnclampedKnots,

    /// The knot vector must be normalized to the domain [0, 1].
    #[error("the knot vector must be normalized to the domain [0, 1]")]
    UnnormalizedKnots,

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

    /// The merge cannot keep every constrained point, for example the end of the left curve and the start of
    /// the right curve when the two points differ.
    #[error("the constraints contradict each other, so the merged curve cannot keep every constrained point")]
    ConflictingConstraints,

    /// The penalization strength must be finite and not negative.
    #[error("the penalization strength {strength} must be finite and not negative")]
    InvalidPenalizationStrength {
        /// The offending penalization strength.
        strength: f64,
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

    /// The penalty needs at least one difference: the difference order must be smaller than the number
    /// of control points that the fit places, n + 1 with loose ends and n − 1 with fixed ends.
    #[error(
        "the difference order {difference_order} of the penalization must be smaller than \
         the {free_control_points} control points that the fit places"
    )]
    DifferenceOrderTooLarge {
        /// The offending difference order.
        difference_order: usize,
        /// The number of control points that the fit places.
        free_control_points: usize,
    },

    /// Penalized fitting requires a uniform knot vector (see `Eilers1996`).
    #[error("penalized fitting requires a uniform knot vector")]
    NonUniformKnots,

    /// Parameters need at least two data points: the first gets the parameter 0, the last gets 1.
    #[error("the data holds {count} points, but at least 2 are needed")]
    TooFewDataPoints {
        /// The number of data points.
        count: usize,
    },

    /// A data point has a coordinate that is NaN or infinite.
    #[error("the data point at index {index} has a coordinate that is not finite")]
    NonFiniteDataPoint {
        /// The index of the data point.
        index: usize,
    },

    /// The chord-length and centripetal methods need data points at two or more positions.
    #[error("all data points lie at the same position, so their chord lengths give no parameters")]
    CoincidentDataPoints,

    /// Parameters must be in non-decreasing order.
    #[error("the parameter at index {index} is smaller than the parameter before it")]
    DecreasingParameters {
        /// The index of the smaller parameter.
        index: usize,
    },

    /// The interpolation system is singular or nearly singular, so no curve passes through all data points: a basis
    /// function is zero or almost zero at its own parameter (the Schoenberg-Whitney condition). Uniform knots with
    /// uneven parameters cause this.
    #[error(
        "the interpolation system is singular at the data point {index}, so no curve passes through all data \
         points; choose other parameter or knot methods"
    )]
    SingularInterpolation {
        /// The index of the data point that the curve cannot pass through.
        index: usize,
    },

    /// A calculation left the range of `f64` values, for example because the input magnitudes are too large.
    #[error("a calculation produced a value that is not finite; reduce the magnitude of the input")]
    NonFiniteValue,
}
