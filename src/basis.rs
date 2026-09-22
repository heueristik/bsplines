//! Implements the basis functions by the Cox-de Boor-Mansfield recurrence.

use nalgebra::DVector;

/// Evaluates the `i`-th basis function of degree `p` at the parameter `u`
/// by the Cox-de Boor-Mansfield recurrence — see [`Knots::basis`](crate::knots::Knots::basis).
/// It evaluates the recurrence from degree 0 upward, so the time grows with p² and not with 2ᵖ.
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
    let last = polygon_segments - derivative;

    // A stack buffer holds the values for the common low degrees, so they need no allocation.
    let mut stack_buffer = [0.0; 16];
    let mut heap_buffer = Vec::new();
    let values: &mut [f64] = if degree < stack_buffer.len() {
        &mut stack_buffer[..=degree]
    } else {
        heap_buffer.resize(degree + 1, 0.0);
        &mut heap_buffer
    };

    // The basis functions of degree 0 from the index i to i + p.
    for (offset, value) in values.iter_mut().enumerate() {
        let j = index + offset;
        let is_in_interval = knots[j] <= u && u < knots[j + 1];
        let closes_last_interval = j == last && u == knots[last + 1];
        *value = if is_in_interval || closes_last_interval { 1.0 } else { 0.0 };
    }

    // Each level raises the degree by one and keeps one basis function fewer.
    for level in 1..=degree {
        for offset in 0..=degree - level {
            let j = index + offset;

            let summand1 = if knots[j + level] == knots[j] {
                0.0
            } else {
                (u - knots[j]) / (knots[j + level] - knots[j]) * values[offset]
            };

            let summand2 = if knots[j + level + 1] == knots[j + 1] {
                0.0
            } else {
                // This form is numerically more stable than the algebraically equal
                // `(1.0 - (u - knots[j + 1]) / (knots[j + level + 1] - knots[j + 1])) * values[offset + 1]`.
                (knots[j + level + 1] - u) / (knots[j + level + 1] - knots[j + 1]) * values[offset + 1]
            };

            values[offset] = summand1 + summand2;
        }
    }
    values[0]
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
    fn basis_functions_of_degree_40_sum_to_one() {
        let degree = 40;
        let knots = Knots::uniform(degree, degree + 5).unwrap();

        for u in [0.0, 0.3, 0.7, 1.0] {
            let sum: f64 = (0..=knots.polygon_segments()).map(|index| knots.basis(index, u).unwrap()).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-9);
        }
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
