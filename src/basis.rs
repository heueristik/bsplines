//! Implements the basis functions by the Cox-de Boor-Mansfield recurrence.

use nalgebra::DVector;

use crate::buffer::with_buffer;

/// Evaluates the `i`-th basis function of degree `p` at the parameter `u`
/// by the Cox-de Boor-Mansfield recurrence — see [`Knots::basis`](crate::knots::Knots::basis).
/// It evaluates the recurrence from degree 0 upward, so the time grows with p² and not with 2ᵖ.
///
/// The last basis function also covers the end of the domain, so it is 1 at the last knot.
pub(crate) fn basis(knots: &DVector<f64>, index: usize, degree: usize, u: f64) -> f64 {
    // A knot vector of n + p + 2 knots has the basis functions 0 to n.
    let last = knots.len() - degree - 2;

    with_buffer(degree + 1, |values| {
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

                // A zero value adds nothing, and skipping it keeps an infinite ratio of very close knots from
                // turning the product into NaN.
                let summand1 = if knots[j + level] == knots[j] || values[offset] == 0.0 {
                    0.0
                } else {
                    (u - knots[j]) / (knots[j + level] - knots[j]) * values[offset]
                };

                let summand2 = if knots[j + level + 1] == knots[j + 1] || values[offset + 1] == 0.0 {
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
    })
}

/// Evaluates the p + 1 basis functions of degree `p` that are not zero in the knot span `span` at the parameter `u`
/// by the recurrence of [`Knots::basis`](crate::knots::Knots::basis) — algorithm A2.2 in `Piegl1997`. The value at
/// the offset r belongs to the basis function span − p + r. Each degree reuses the values of the degree below for all
/// functions together, so the time grows with p².
///
/// The span must be a knot span from [`Knots::find_span`](crate::knots::Knots::find_span), so no knot interval
/// in the recurrence has zero length.
pub(crate) fn calculate_basis_values(knots: &DVector<f64>, span: usize, degree: usize, u: f64, values: &mut [f64]) {
    debug_assert_eq!(values.len(), degree + 1, "a span has p + 1 basis functions that are not zero");
    values[0] = 1.0;

    // Each level raises the degree by one and adds one basis function.
    for level in 1..=degree {
        // The first summand of the next basis function comes from the same basis function of the degree below.
        let mut first_summand = 0.0;
        for offset in 0..level {
            let lower_value = values[offset];
            let left_knot = knots[span + offset + 1 - level];
            let right_knot = knots[span + offset + 1];

            values[offset] = first_summand + (right_knot - u) / (right_knot - left_knot) * lower_value;
            first_summand = (u - left_knot) / (right_knot - left_knot) * lower_value;
        }
        values[level] = first_summand;
    }
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
    fn basis_functions_of_very_close_knots_sum_to_one() {
        let knots = Knots::new(2, dvector![0., 0., 0., 1e-310, 1., 1., 1.]).unwrap();
        assert!((1.0 / 1e-310_f64).is_infinite(), "the ratio over the closest knots overflows");

        let sum: f64 = (0..=knots.polygon_segments()).map(|index| knots.basis(index, 0.5).unwrap()).sum();
        assert_relative_eq!(sum, 1.0, epsilon = f64::EPSILON.sqrt());
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
        let derivative_knots = knots.derivative_knots(derivative).unwrap();
        let basis = |index, u| derivative_knots.basis(index, u).unwrap();

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
}
