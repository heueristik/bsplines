//! Implements the basis functions by the Cox-de Boor-Mansfield recurrence.

use nalgebra::DVector;

/// Evaluates the `i`-th basis function of degree `p` at the parameter `u`
/// by the Cox-de Boor-Mansfield recurrence — see [`Knots::basis`](crate::knots::Knots::basis).
///
/// The derivative order `k` and the number of polygon segments `n` close the
/// last interval, so the last basis function covers `u = 1`.
pub(crate) fn basis(
    knots: &DVector<f64>,
    index: usize,
    degree: usize,
    derivative: usize,
    polygon_segments: usize,
    u: f64,
) -> f64 {
    if degree == 0 {
        if (knots[index] <= u && u < knots[index + 1]) ||
            (index == polygon_segments - derivative && u == knots[polygon_segments + 1 - derivative])
        {
            return 1.0;
        }
        return 0.0;
    }

    let summand1 = if knots[index + degree] == knots[index] {
        0.0
    } else {
        (u - knots[index]) / (knots[index + degree] - knots[index]) *
            basis(knots, index, degree - 1, derivative, polygon_segments, u)
    };

    let summand2 = if knots[index + degree + 1] == knots[index + 1] {
        0.0
    } else {
        // This form is numerically more stable than the algebraically equal
        // `(1.0 - (u - knots[index + 1]) / (knots[index + degree + 1] - knots[index + 1])) * basis(…)`.
        (knots[index + degree + 1] - u) / (knots[index + degree + 1] - knots[index + 1]) *
            basis(knots, index + 1, degree - 1, derivative, polygon_segments, u)
    };

    summand1 + summand2
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    use crate::knots::Knots;

    #[test]
    fn basis_functions_degree_3() {
        let degree = 3;
        let knots = Knots::new(degree, dvector![0., 0., 0., 0., 1. / 3., 2. / 3., 1., 1., 1., 1.]).unwrap();
        let basis = |index, u| knots.basis(index, u).unwrap();

        let mut i = 0;
        assert_eq!(basis(i, 0.0), 1.0);
        assert_eq!(basis(i, 1. / 6.), 1. / 8.);
        assert_eq!(basis(i, 1. / 3.), 0.0);
        assert_eq!(basis(i, 1. / 2.), 0.0);
        assert_eq!(basis(i, 2. / 3.), 0.0);
        assert_eq!(basis(i, 5. / 6.), 0.0);
        assert_eq!(basis(i, 1.), 0.0);

        i = 1;
        assert_eq!(basis(i, 0.), 0.0);
        assert_eq!(basis(i, 1. / 6.), 19. / 32.);
        assert_eq!(basis(i, 1. / 3.), 1. / 4.);
        assert_relative_eq!(basis(i, 1. / 2.), 1. / 32., epsilon = f64::EPSILON.sqrt());
        assert_eq!(basis(i, 2. / 3.), 0.0);
        assert_eq!(basis(i, 5. / 6.), 0.0);
        assert_eq!(basis(i, 1.), 0.0);

        i = 2;
        assert_eq!(basis(i, 0.), 0.0);
        assert_eq!(basis(i, 1. / 6.), 25. / 96.);
        assert_eq!(basis(i, 1. / 3.), 7. / 12.);
        assert_relative_eq!(basis(i, 1. / 2.), 15. / 32., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(basis(i, 2. / 3.), 1. / 6., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(basis(i, 5. / 6.), 1. / 48., epsilon = f64::EPSILON.sqrt());
        assert_eq!(basis(i, 1.0), 0.0);

        i = 3;
        assert_eq!(basis(i, 0.), 0.0);
        assert_eq!(basis(i, 1. / 6.), 1. / 48.);
        assert_eq!(basis(i, 1. / 3.), 1. / 6.);
        assert_relative_eq!(basis(i, 1. / 2.), 15. / 32., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(basis(i, 2. / 3.), 7. / 12., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(basis(i, 5. / 6.), 25. / 96., epsilon = f64::EPSILON.sqrt());
        assert_eq!(basis(i, 1.0), 0.0);

        i = 4;
        assert_eq!(basis(i, 0.), 0.0);
        assert_eq!(basis(i, 1. / 6.), 0.0);
        assert_eq!(basis(i, 1. / 3.), 0.0);
        assert_relative_eq!(basis(i, 1. / 2.), 1. / 32., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(basis(i, 2. / 3.), 1. / 4., epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(basis(i, 5. / 6.), 19. / 32., epsilon = f64::EPSILON.sqrt());
        assert_eq!(basis(i, 1.0), 0.0);

        i = 5;
        assert_eq!(basis(i, 0.0), 0.0);
        assert_eq!(basis(i, 1. / 6.), 0.);
        assert_eq!(basis(i, 1. / 3.), 0.0);
        assert_eq!(basis(i, 1. / 2.), 0.0);
        assert_eq!(basis(i, 2. / 3.), 0.0);
        assert_relative_eq!(basis(i, 5. / 6.), 1. / 8., epsilon = f64::EPSILON.sqrt());
        assert_eq!(basis(i, 1.), 1.0);
    }

    #[test]
    fn basis_functions_degree_4_derivative_1() {
        let derivative = 1;
        let degree = 4;
        let knots = Knots::new(degree, dvector![0., 0., 0., 0., 0., 1. / 3., 2. / 3., 1., 1., 1., 1., 1.]).unwrap();

        let mut i = 0;
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 0.0), 1.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 6.), 1. / 8.);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 3.), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 2.), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 2. / 3.), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 5. / 6.), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1.), 0.0);

        i = 1;
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 0.), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 6.), 19. / 32.);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 3.), 1. / 4.);
        assert_relative_eq!(
            knots.basis_of_derivative_curve(derivative, i, 1. / 2.),
            1. / 32.,
            epsilon = f64::EPSILON.sqrt()
        );
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 2. / 3.), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 5. / 6.), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1.), 0.0);

        i = 2;
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 0.), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 6.), 25. / 96.);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 3.), 7. / 12.);
        assert_relative_eq!(
            knots.basis_of_derivative_curve(derivative, i, 1. / 2.),
            15. / 32.,
            epsilon = f64::EPSILON.sqrt()
        );
        assert_relative_eq!(
            knots.basis_of_derivative_curve(derivative, i, 2. / 3.),
            1. / 6.,
            epsilon = f64::EPSILON.sqrt()
        );
        assert_relative_eq!(
            knots.basis_of_derivative_curve(derivative, i, 5. / 6.),
            1. / 48.,
            epsilon = f64::EPSILON.sqrt()
        );
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1.0), 0.0);

        i = 3;
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 0.), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 6.), 1. / 48.);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 3.), 1. / 6.);
        assert_relative_eq!(
            knots.basis_of_derivative_curve(derivative, i, 1. / 2.),
            15. / 32.,
            epsilon = f64::EPSILON.sqrt()
        );
        assert_relative_eq!(
            knots.basis_of_derivative_curve(derivative, i, 2. / 3.),
            7. / 12.,
            epsilon = f64::EPSILON.sqrt()
        );
        assert_relative_eq!(
            knots.basis_of_derivative_curve(derivative, i, 5. / 6.),
            25. / 96.,
            epsilon = f64::EPSILON.sqrt()
        );
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1.0), 0.0);

        i = 4;
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 0.), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 6.), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 3.), 0.0);
        assert_relative_eq!(
            knots.basis_of_derivative_curve(derivative, i, 1. / 2.),
            1. / 32.,
            epsilon = f64::EPSILON.sqrt()
        );
        assert_relative_eq!(
            knots.basis_of_derivative_curve(derivative, i, 2. / 3.),
            1. / 4.,
            epsilon = f64::EPSILON.sqrt()
        );
        assert_relative_eq!(
            knots.basis_of_derivative_curve(derivative, i, 5. / 6.),
            19. / 32.,
            epsilon = f64::EPSILON.sqrt()
        );
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1.0), 0.0);

        i = 5;
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 0.0), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 6.), 0.);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 3.), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1. / 2.), 0.0);
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 2. / 3.), 0.0);
        assert_relative_eq!(
            knots.basis_of_derivative_curve(derivative, i, 5. / 6.),
            1. / 8.,
            epsilon = f64::EPSILON.sqrt()
        );
        assert_eq!(knots.basis_of_derivative_curve(derivative, i, 1.), 1.0);
    }
}
