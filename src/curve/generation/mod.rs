#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("points", "doc-images/plots/generation/points.svg"),
doc = ::embed_doc_image::embed_image!("manual", "doc-images/plots/generation/manual.svg"),
doc = ::embed_doc_image::embed_image!("interpolation", "doc-images/plots/generation/interpolation.svg"),
doc = ::embed_doc_image::embed_image!("fit-loose-all", "doc-images/plots/generation/fit-loose-all.svg"),
doc = ::embed_doc_image::embed_image!("fit-loose-half", "doc-images/plots/generation/fit-loose-half.svg"),
doc = ::embed_doc_image::embed_image!("fit-loose-half-penalized", "doc-images/plots/generation/fit-loose-half-penalized.svg")))]
//! Generates a curve with different methods knot into the curve.
//!
//! ## Methods
//!
//! - Manual control polygon
//! - Interpolation
//! - Least-Squares Fit
//!   - fixed start and end points
//!   - unpenalized/penalized
//!
//!
//! | Raw Data | Manual Control Polygon | Interpolation |F
//! |:------------------------------|:--------------------------|:--------------------------|
//! | ![][points]             | ![][manual]                            | ![][interpolation] |
//! | Scattered, 2-dimensional data points<br>(`N = 18`).<br><br>   | Curve of degree `p = 2` with `n = N-1`<br>segments and control points generated<br>directly from the data points. | Curve of degree `p = 2` and `n = N-1`<br>segments interpolating the data points.<br><br> |
//! |  <br> | |
//! | Least-Squares Fit  | Least-Squares Fit | Penalized Least-Squares Fit |
//! | ![][fit-loose-all]             | ![][fit-loose-half]        | ![][fit-loose-half-penalized]   |
//! | Curve of degree `p = 2` with `n = N-1`<br>segments approximating the data points.<br> | Curve of degree `p = 2` with `n= N/3`<br>segments approximating the data points.<br> | Curve of degree `p = 2` with `n= N/3`<br>segments approximating the data<br> points penalized with `λ = 1`, `κ = 2`. |

use crate::curve::{
    knots, parameters,
    points::{
        methods::{fit, fit::Penalization, interpolation},
        ControlPoints, DataPoints, Points,
    },
    Curve, CurveError,
};

#[derive()]
pub enum Generation<'a> {
    Manual {
        degree: usize,
        points: ControlPoints,
        knots: knots::Generation,
    },
    Interpolation {
        degree: usize,
        points: &'a DataPoints,
    },
    LeastSquaresFit {
        degree: usize,
        points: &'a DataPoints,
        intended_segments: usize,
        method: fit::Method,
        penalization: Option<Penalization>,
    },
}

/// Returns a B-Spline
///
/// # Arguments
///
/// * `degree` - The degree of the spline
///
/// # Examples
/// ```
/// use nalgebra::dmatrix;
/// use bsplines::curve::generation::{generate, Generation::Manual};
/// use bsplines::curve::knots::Generation::Uniform;
/// use bsplines::curve::points::ControlPoints;
///
/// // Create a coordinate matrix containing with four 2D points.
/// let points = ControlPoints::new(dmatrix![
/// // 1    2    3    4    5
///  -2.0,-2.0,-1.0, 0.5, 1.5; // x
///  -1.0, 0.0, 1.0, 1.0, 2.0; // y
/// ]);
/// let degree = 2;
/// let curve = generate(Manual{degree, points, knots: Uniform}).unwrap();
/// println!("{:?}", curve.evaluate(0.5));
/// ```
pub fn generate(generation: Generation) -> Result<Curve, CurveError> {
    match generation {
        Generation::Manual { degree, points, knots: method } => {
            // TODO more sanity checks
            let n = points.segments();
            let p = degree;

            if points.segments() < degree {
                return Err(CurveError::DegreeAndSegmentsMismatch { p, n });
            }

            let data_points = DataPoints::new(points.matrix().clone());
            let knots = match method {
                knots::Generation::Uniform => knots::methods::uniform(p, n),
                knots::Generation::Manual { knots } => Ok(knots.clone()),
                knots::Generation::Method {
                    // TODO default methods deBoor + chord length
                    parameter_method,
                    knot_method,
                } => {
                    let params = parameters::generate(&data_points, parameter_method);
                    knots::generate(p, n, &params, knot_method)
                }
            };
            Curve::new(knots?, points)
        }
        Generation::Interpolation { degree, points } => {
            // TODO allow other methods + uniform
            let params = parameters::generate(points, parameters::Method::EquallySpaced); // TODO Piegl: ChordLength + DeBoor
            let knots = knots::generate(degree, points.segments(), &params, knots::Method::Uniform)?;
            let points = ControlPoints::new_with_capacity(
                interpolation::interpolate(&knots, points, &params),
                knots.degree() + 1,
            );

            Curve::new(knots, points)
        }
        Generation::LeastSquaresFit { degree, points, intended_segments, method, penalization } => {
            // TODO allow other methods + uniform
            let params = parameters::generate(points, parameters::Method::EquallySpaced);
            let knots = knots::generate(degree, intended_segments, &params, knots::Method::Uniform)?;

            /*let (knots, params) = match penalization {
                Some(..) => {
                    let params = parameters::generate(&points, parameters::Method::Centripetal);
                    let knots = knots::generate(degree, segments, &params, knots::Method::DeBoor)?;
                    (knots, params)
                }
                None => {
                    let params = parameters::generate(points, parameters::Method::EquallySpaced);
                    let knots = knots::generate(degree, segments, &params, knots::Method::Uniform)?;
                    (knots, params)
                }
            };*/

            let points = match method {
                fit::Method::FixedEnds => ControlPoints::new_with_capacity(
                    fit::fixed::fit(&knots, points, &params, penalization)
                        .map_err(|err| CurveError::FitError { err })?,
                    degree + 1,
                ),
                fit::Method::LooseEnds => ControlPoints::new_with_capacity(
                    fit::loose::fit(&knots, points, &params, penalization)
                        .map_err(|err| CurveError::FitError { err })?,
                    degree + 1,
                ),
            };
            Curve::new(knots, points)
        }
    }
}
