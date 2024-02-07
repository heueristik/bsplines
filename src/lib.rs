//#![warn(missing_docs)]
//#![warn(missing_doc_code_examples)]
#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("split-after", "doc-images/plots/manipulation/split-after.svg")))]
//! `bsplines` is a Rust library for vectorized, N-dimensional B-spline curves and their derivatives based on
//! [nalgebra].
//!
//! ## Features
//! - Create `N`-dimensional (`x = 1, 2, 3,...`) curves of arbitrary polynomial degree `p`.
//! - Efficient [curve evaluation][curve::Curve] for all available derivatives `k = 0, 1,... , p`.
//! - Built with [nalgebra](https://crates.io/crates/nalgebra) to store point data in contiguous arrays
//! - Multiple methods for
//!   - [curve generation][curve::generation]
//!   - [curve parametrization][curve::parameters]
//!   - [knot generation][curve::knots]
//!   - [curve manipulation][manipulation]
//!     - [knot insertion][manipulation::insert]
//!     - [reversing][manipulation::reverse]
//!     - [splitting][manipulation::split]
//!     - [merging][manipulation::split]
//!
//! ## Mathematical Definition
//!
//! B-splines are parametric functions composed of piecewise polynomials with a polynomial degree `p > 0`.
//! These piecewise polynomials are joined so that the parametric function is `p-1` times continuously
//! differentiable. The overall functions are parametrized over finite domains with the co-domain being an
//! `N`-dimensional vector space. They can describe curves, but also surfaces.
//!
//! A B-spline curve can be defined by
//! ```math
//! \mathcal{C}(u) = \sum_{i=0}^{n} \mathcal{N}_{i,p}^{\boldsymbol{U}} (u)\, \boldsymbol{P}_i
//! ```
//! with
//! `u`, a parameter on the curve, usually limited to `u ∈ [0,1]`.
//! `n` the number of polynomial spline segments
//! `p`, the spline degree
//! `U`, the knot vector
//! `N`, the `n + 1` spline basis functions of degree `p`
//! `P`, the `n + 1` control points that can be of arbitrary dimension
//!
//! These characteristics lead to many desirable properties.
//! The piecewise definition makes B-spline functions versatile allowing to interpolate or approximate
//! complex-shaped and high-dimensional data, while maintaining a low polynomial degree. Because of the polynomial
//! nature, all possible derivatives are accessible.
//!
//! ![A 2D B-Spline curve.][split-after]
//!
//! Still, evaluations or spatial manipulations can be executed fast because only local polynomial segments must be
//! considered and the associated numerical procedures are stable. Lastly, polynomials represent a memory-efficient way
//! of storing spatial information as few polynomial coefficients suffice to describe complex shapes.

//! ## Literature:
//! |            |                                                                                                                                                                    |
//! |-----------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
//! | Piegl1997  | Piegl, L., Tiller, W. The NURBS Book. Monographs in Visual Communication. Springer, Berlin, Heidelberg, 2nd ed., 1997.                                             |
//! | Eilers1996 | Eilers, P. H. C., Marx, B. D., Flexible smoothing with B -splines and penalties, Stat. Sci., 11(2) (1996) 89–121.                                                  |
//! | Tai2003    | Tai, C.-L., Hu, S.-M., Huang, Q.-X., Approximate merging of B-spline curves via knot adjustment and constrained optimization, Comput. Des., 35(10) (2003) 893–899. |

pub mod curve;
pub mod manipulation;
pub mod types;
