#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("reverse-before", "doc-images/plots/manipulation/reverse-before.svg"),
doc = ::embed_doc_image::embed_image!("reverse-after", "doc-images/plots/manipulation/reverse-after.svg")))]
//! Reverses the curve parametrization.
//!
//! | A normal curve.      | The reversed curve. |
//! |:--------------------:|:-------------------:|
//! | ![][reverse-before]  | ![][reverse-after]  |

use crate::curve::Curve;

pub fn reverse(curve: &mut Curve) -> &mut Curve {
    curve.knots.reverse();
    curve.points.reverse();
    curve
}

// TODO
pub fn reversed(curve: &Curve) -> Curve {
    let mut clone = curve.clone();
    clone.reverse();
    clone
}
