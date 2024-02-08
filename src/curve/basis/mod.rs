#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("eq-basis-function", "doc-images/equations/basis-function.svg"),
doc = ::embed_doc_image::embed_image!("eq-basis-prefactor", "doc-images/equations/basis-prefactor.svg"),
doc = ::embed_doc_image::embed_image!("eq-basis-function-zero", "doc-images/equations/basis-function-zero.svg")))]
//! Evaluates the basis spline functions using the Cox-de Boor-Mansfield recurrence relation
//!
//! ![The Cox-de Boor-Mansfield recurrence relation][eq-basis-function]
//!
//! with the basis functions of degree `p = 0`
//!
//! ![Basis function of degree zero][eq-basis-function-zero]
//!
//! where the conditional `⋁ (i = n - k ⋀ u = U_{n+1-k)` closes the last interval
//! and the pre-factors
//!
//! ![Pre-factors][eq-basis-prefactor]

use crate::types::VecD;

/// Evaluates the `i`-th basis spline function of degree `p`
///
/// ## Arguments
///
/// - `i` the index with `i ∈ {0, 1, ..., n}`
/// - `p` the spline degree
/// - `k` the derivative order
/// - `U` the knot vector
pub fn basis(Uk: &VecD, i: usize, p: usize, k: usize, n: usize, u: f64) -> f64 {
    if p == 0 {
        if (Uk[i] <= u && u < Uk[i + 1]) || (i == n - k && u == Uk[n + 1 - k]) {
            return 1.0;
        }
        return 0.0;
    }

    let summand1 = if Uk[i + p] == Uk[i] {
        0.0
    } else {
        let g = i;
        let h = p - 1;
        (u - Uk[g]) / (Uk[g + h + 1] - Uk[g]) * basis(Uk, i, h, k, n, u)
    };

    let summand2 = if Uk[i + 1 + p] == Uk[i + 1] {
        0.0
    } else {
        let g = i + 1;
        let h = p - 1;

        // The following equation is numerically more stable than
        // `(1.0 - ((u - Uk[g]) / (Uk[g + h + 1] - Uk[g]))) * self.evaluate(k, g, h, u)`
        (Uk[g + p] - u) / (Uk[g + h + 1] - Uk[g]) * basis(Uk, g, h, k, n, u)
    };

    summand1 + summand2
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    use crate::curve::knots::Knots;

    const SEGMENTS: usize = 4;

    #[test]
    fn basis_func_degree3() {
        let k = 0;
        let p = 3;
        let knots = Knots::new(p, dvector![0., 0., 0., 0., 1. / 3., 2. / 3., 1., 1., 1., 1.]);

        // Basis function i = 0
        let mut i = 0;
        assert_eq!(knots.evaluate(k, i, p, 0.0), 1.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 6.), 1. / 8.);
        assert_eq!(knots.evaluate(k, i, p, 1. / 3.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 2.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 2. / 3.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 5. / 6.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1.), 0.0);

        i = 1;
        assert_eq!(knots.evaluate(k, i, p, 0.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 6.), 19. / 32.);
        assert_eq!(knots.evaluate(k, i, p, 1. / 3.), 1. / 4.);
        assert_relative_eq!(knots.evaluate(k, i, p, 1. / 2.), 1. / 32., epsilon = f64::EPSILON.sqrt());
        assert_eq!(knots.evaluate(k, i, p, 2. / 3.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 5. / 6.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1.), 0.0);

        i = 2;
        assert_eq!(knots.evaluate(k, i, p, 0.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 6.), 25. / 96.);
        assert_eq!(knots.evaluate(k, i, p, 1. / 3.), 7. / 12.);
        assert_relative_eq!(knots.evaluate(k, i, p, 1. / 2.), 15. / 32., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(knots.evaluate(k, i, p, 2. / 3.), 1. / 6., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(knots.evaluate(k, i, p, 5. / 6.), 1. / 48., epsilon = f64::EPSILON.sqrt());
        assert_eq!(knots.evaluate(k, i, p, 1.0), 0.0);

        i = 3;
        assert_eq!(knots.evaluate(k, i, p, 0.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 6.), 1. / 48.);
        assert_eq!(knots.evaluate(k, i, p, 1. / 3.), 1. / 6.);
        assert_relative_eq!(knots.evaluate(k, i, p, 1. / 2.), 15. / 32., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(knots.evaluate(k, i, p, 2. / 3.), 7. / 12., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(knots.evaluate(k, i, p, 5. / 6.), 25. / 96., epsilon = f64::EPSILON.sqrt());
        assert_eq!(knots.evaluate(k, i, p, 1.0), 0.0);

        i = 4;
        assert_eq!(knots.evaluate(k, i, p, 0.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 6.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 3.), 0.0);
        assert_relative_eq!(knots.evaluate(k, i, p, 1. / 2.), 1. / 32., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(knots.evaluate(k, i, p, 2. / 3.), 1. / 4., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(knots.evaluate(k, i, p, 5. / 6.), 19. / 32., epsilon = f64::EPSILON.sqrt());
        assert_eq!(knots.evaluate(k, i, p, 1.0), 0.0);

        i = 5;
        assert_eq!(knots.evaluate(k, i, p, 0.0), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 6.), 0.);
        assert_eq!(knots.evaluate(k, i, p, 1. / 3.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 2.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 2. / 3.), 0.0);
        assert_relative_eq!(knots.evaluate(k, i, p, 5. / 6.), 1. / 8., epsilon = f64::EPSILON.sqrt());
        assert_eq!(knots.evaluate(k, i, p, 1.), 1.0);
    }

    #[test]
    fn basis_func_degree4() {
        let k = 1;
        let p = 4;
        let knots = Knots::new(p, dvector![0., 0., 0., 0., 0., 1. / 3., 2. / 3., 1., 1., 1., 1., 1.]);

        // Basis function i = 0
        let mut i = 0;
        assert_eq!(knots.evaluate(k, i, p, 0.0), 1.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 6.), 1. / 8.);
        assert_eq!(knots.evaluate(k, i, p, 1. / 3.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 2.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 2. / 3.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 5. / 6.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1.), 0.0);

        i = 1;
        assert_eq!(knots.evaluate(k, i, p, 0.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 6.), 19. / 32.);
        assert_eq!(knots.evaluate(k, i, p, 1. / 3.), 1. / 4.);
        assert_relative_eq!(knots.evaluate(k, i, p, 1. / 2.), 1. / 32., epsilon = f64::EPSILON.sqrt());
        assert_eq!(knots.evaluate(k, i, p, 2. / 3.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 5. / 6.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1.), 0.0);

        i = 2;
        assert_eq!(knots.evaluate(k, i, p, 0.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 6.), 25. / 96.);
        assert_eq!(knots.evaluate(k, i, p, 1. / 3.), 7. / 12.);
        assert_relative_eq!(knots.evaluate(k, i, p, 1. / 2.), 15. / 32., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(knots.evaluate(k, i, p, 2. / 3.), 1. / 6., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(knots.evaluate(k, i, p, 5. / 6.), 1. / 48., epsilon = f64::EPSILON.sqrt());
        assert_eq!(knots.evaluate(k, i, p, 1.0), 0.0);

        i = 3;
        assert_eq!(knots.evaluate(k, i, p, 0.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 6.), 1. / 48.);
        assert_eq!(knots.evaluate(k, i, p, 1. / 3.), 1. / 6.);
        assert_relative_eq!(knots.evaluate(k, i, p, 1. / 2.), 15. / 32., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(knots.evaluate(k, i, p, 2. / 3.), 7. / 12., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(knots.evaluate(k, i, p, 5. / 6.), 25. / 96., epsilon = f64::EPSILON.sqrt());
        assert_eq!(knots.evaluate(k, i, p, 1.0), 0.0);

        i = 4;
        assert_eq!(knots.evaluate(k, i, p, 0.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 6.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 3.), 0.0);
        assert_relative_eq!(knots.evaluate(k, i, p, 1. / 2.), 1. / 32., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(knots.evaluate(k, i, p, 2. / 3.), 1. / 4., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(knots.evaluate(k, i, p, 5. / 6.), 19. / 32., epsilon = f64::EPSILON.sqrt());
        assert_eq!(knots.evaluate(k, i, p, 1.0), 0.0);

        i = 5;
        assert_eq!(knots.evaluate(k, i, p, 0.0), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 6.), 0.);
        assert_eq!(knots.evaluate(k, i, p, 1. / 3.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 1. / 2.), 0.0);
        assert_eq!(knots.evaluate(k, i, p, 2. / 3.), 0.0);
        assert_relative_eq!(knots.evaluate(k, i, p, 5. / 6.), 1. / 8., epsilon = f64::EPSILON.sqrt());
        assert_eq!(knots.evaluate(k, i, p, 1.), 1.0);
    }
}
