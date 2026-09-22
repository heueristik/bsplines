#![warn(missing_docs)]
#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("eq-curve", "doc-images/equations/curve.svg"),
doc = ::embed_doc_image::embed_image!("eq-knots", "doc-images/equations/knots.svg"),
doc = ::embed_doc_image::embed_image!("eq-control-points", "doc-images/equations/control-points.svg"),
doc = ::embed_doc_image::embed_image!("img-curve", "doc-images/plots/manipulation/insert-before.svg")))]
//! **bsplines** is a library for vectorized, N-dimensional B-spline curves and their derivatives based on
//! [nalgebra].
//!
//! ## Features
//! - Create `N`-dimensional (`N = 1, 2, 3,...`) curves of arbitrary polynomial degree `p`.
//! - Efficient [curve evaluation][Curve::evaluate] for all available derivatives `k = 0, 1,... , p`.
//! - Built with [nalgebra](https://crates.io/crates/nalgebra) to store point data in contiguous arrays
//! - Multiple methods for
//!   - [curve generation][Curve]
//!   - [curve parametrization][ParameterMethod]
//!   - [knot generation][KnotMethod]
//!   - curve manipulation
//!     - [knot insertion][Curve::insert_knot]
//!     - [reversing][Curve::reverse]
//!     - [splitting][Curve::split]
//!     - [merging][Curve::append]
//!
//! ## Example
//! ```
//! use bsplines::{Curve, DataPoints};
//! use nalgebra::dmatrix;
//!
//! # fn main() -> bsplines::Result<()> {
//! // Five 2D data points, one column per point.
//! let data = DataPoints::new(dmatrix![
//!     -2.0,-1.0, 0.0, 1.0, 2.0; // x
//!      0.5,-0.5, 1.5,-1.5, 0.5; // y
//! ]);
//!
//! // Interpolate the data with a cubic curve and evaluate it.
//! let curve = Curve::interpolate(&data, 3)?;
//! let point = curve.evaluate(0.5)?;
//! let velocity = curve.evaluate_derivative(0.5, 1)?;
//!
//! // Or approximate it with a penalized least-squares fit.
//! let fitted = Curve::fit(&data, 3).loose_ends().penalized(0.5, 2).build()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## What are B-Splines?
//!
//! B-splines are parametric functions composed of piecewise, polynomial [basis functions][Knots::basis] of degree `p >
//! 0`. These piecewise polynomials are joined so that the parametric function is `p-1` times continuously
//! differentiable. The overall functions are parametrized over finite domains with a so-called [knot
//! vector][Knots] with the co-domain being an `N`-dimensional vector space, that is defined by [control
//! points][ControlPoints]. They can describe [curves][Curve], but also surfaces.

//! These characteristics lead to many desirable properties.
//! The piecewise definition makes B-spline functions versatile allowing to interpolate or approximate
//! complex-shaped and high-dimensional data, while maintaining a low polynomial degree. Because of the polynomial
//! nature, all possible derivatives are accessible.
//!
//! ![A 2D B-Spline curve.][img-curve]
//!
//! Still, evaluations or spatial manipulations can be executed fast because only local polynomial segments must be
//! considered and the associated numerical procedures are stable.
//! Lastly, polynomials represent a memory-efficient way of storing spatial information
//! as few polynomial coefficients suffice to describe complex shapes.

//! ## Literature:
//! |            |                                                                                                                                                                    |
//! |-----------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
//! | Piegl1997  | Piegl, L., Tiller, W. The NURBS Book. Monographs in Visual Communication. Springer, Berlin, Heidelberg, 2nd ed., 1997.                                             |
//! | Eilers1996 | Eilers, P. H. C., Marx, B. D., Flexible smoothing with B -splines and penalties, Stat. Sci., 11(2) (1996) 89–121.                                                  |
//! | Tai2003    | Tai, C.-L., Hu, S.-M., Huang, Q.-X., Approximate merging of B-spline curves via knot adjustment and constrained optimization, Comput. Des., 35(10) (2003) 893–899. |

mod basis;
mod curve;
mod error;
mod fit;
mod interpolation;
mod knots;
mod manipulation;
mod parameters;
mod points;
mod vector_views;

pub use curve::Curve;
pub use error::{Error, Result};
pub use fit::FitBuilder;
pub use knots::{KnotMethod, Knots};
pub use manipulation::merge::Constraints;
pub use parameters::{ParameterMethod, Parameters};
pub use points::{ControlPoints, DataPoints, Points};
