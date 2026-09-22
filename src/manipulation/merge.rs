//! Merges two curves into one — see `Tai2003`.

use std::ops::{AddAssign, DivAssign, SubAssign};

use nalgebra::{DMatrix, DVector};

use crate::{
    Curve,
    basis::basis,
    buffer::with_buffer,
    error::{Error, Result},
    knots::{Knots, reversed},
    points,
    points::{ControlPoints, Points},
    svd::decompose,
    vector_views::VectorViews,
};

/// The parameters at which the points of two merged curves stay fixed.
///
/// The fields name the side of the joint, for [`Curve::append_constrained`] and
/// [`Curve::prepend_constrained`] alike. Both curves together take fewer than p constraints, each in [0, 1].
///
/// | The end of the left curve fixed.      | The start of the right curve fixed.      |
/// |:-------------------------------------:|:----------------------------------------:|
/// | ![][merge-after-left-end-constrained] | ![][merge-after-right-start-constrained] |
#[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("merge-after-left-end-constrained", "doc-images/plots/manipulation/merge-after-left-end-constrained.svg"))]
#[cfg_attr(feature = "doc-images", doc = ::embed_doc_image::embed_image!("merge-after-right-start-constrained", "doc-images/plots/manipulation/merge-after-right-start-constrained.svg"))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Constraints {
    /// The parameters of the left curve whose points stay fixed.
    pub left: Vec<f64>,
    /// The parameters of the right curve whose points stay fixed.
    pub right: Vec<f64>,
}

impl Constraints {
    fn count(&self) -> usize {
        self.left.len() + self.right.len()
    }
}

/// Merges two curves into one, attaching the end of the left curve to the start
/// of the right one while maintaining continuity of all derivatives — see `Tai2003`.
pub(crate) fn merge(left: &Curve, right: &Curve, constraints: &Constraints) -> Result<Curve> {
    let left_degree = left.degree();
    let right_degree = right.degree();

    if left_degree != right_degree {
        return Err(Error::DegreeMismatch { left: left_degree, right: right_degree });
    }
    if left_degree == 0 {
        return Err(Error::DegreeTooLow { degree: left_degree });
    }

    if left.dimension() != right.dimension() {
        return Err(Error::DimensionMismatch { left: left.dimension(), right: right.dimension() });
    }

    let total_constraints = constraints.count();
    if total_constraints >= left_degree {
        return Err(Error::TooManyConstraints { total: total_constraints, degree: left_degree });
    }
    if let Some(&u) = constraints.left.iter().chain(&constraints.right).find(|u| !(0.0..=1.0).contains(*u)) {
        return Err(Error::OutsideDomain { u, min: 0.0, max: 1.0 });
    }

    let shifts = solve_linear_equation_system(left, right, constraints)?;
    let (left_shifted, right_shifted) = shift_boundary_control_points(left, right, &shifts);

    let (left_adjusted, right_reversed, right_adjusted) = adjust_knot_vectors(left, right);
    let merged_knots = merge_knot_vectors(left, right, &left_adjusted, &right_adjusted);

    let (left_points, right_points) = adjust_control_points_of_both_curves(
        left,
        right,
        &left_shifted,
        &right_shifted,
        &left_adjusted,
        &right_reversed,
        &right_adjusted,
    );

    let merged_points = merge_control_points(left, right, &left_points, &right_points);

    Curve::new(Knots::new(left_degree, merged_knots)?, ControlPoints::new(merged_points))
}

// The names of the block matrices (kv, kw, iv, jw, gv, hw, ipv, jppw, kconst) follow the notation in `Tai2003`.
fn calculate_system_matrix(left: &Curve, right: &Curve, constraints: &Constraints) -> DMatrix<f64> {
    let degree = left.degree();

    let left_constraints = constraints.left.len();
    let right_constraints = constraints.right.len();

    let dimension = 3 * degree + left_constraints + right_constraints;
    let mut system_matrix = DMatrix::zeros(dimension, dimension);

    system_matrix.view_mut((0, 0), (2 * degree, 2 * degree)).copy_from(&DMatrix::identity(2 * degree, 2 * degree));

    system_matrix.view_mut((2 * degree, 0), (degree, degree)).copy_from(&calculate_kv(left));
    system_matrix.view_mut((2 * degree, degree), (degree, degree)).copy_from(&calculate_kw(right));

    system_matrix.view_mut((0, 2 * degree), (degree, degree)).copy_from(&calculate_iv(left));
    system_matrix.view_mut((degree, 2 * degree), (degree, degree)).copy_from(&calculate_jw(right));

    if left_constraints > 0 {
        system_matrix
            .view_mut((3 * degree, 0), (left_constraints, degree))
            .copy_from(&calculate_gv(left, &constraints.left));
        system_matrix
            .view_mut((0, 3 * degree), (degree, left_constraints))
            .copy_from(&calculate_ipv(left, &constraints.left));
    }
    if right_constraints > 0 {
        system_matrix
            .view_mut((3 * degree + left_constraints, degree), (right_constraints, degree))
            .copy_from(&calculate_hw(right, &constraints.right));
        system_matrix
            .view_mut((degree, 3 * degree + left_constraints), (degree, right_constraints))
            .copy_from(&calculate_jppw(right, &constraints.right));
    }

    system_matrix
}

fn calculate_kv(curve: &Curve) -> DMatrix<f64> {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();

    let knot_derivatives = &curve.knots.derivatives;
    let point_matrix = curve.control_points.matrix();

    let mut kv = DMatrix::zeros(degree, degree);

    for derivative in 0..=degree - 1 {
        for i in polygon_segments - degree + 1..=polygon_segments {
            let mut sum = 0.;

            for basis_index in polygon_segments - degree..=polygon_segments - derivative {
                sum += prefactor(degree, basis_index, i, derivative, point_matrix, &knot_derivatives[0]) *
                    basis(
                        &knot_derivatives[derivative],
                        basis_index,
                        degree - derivative,
                        0,
                        polygon_segments - derivative,
                        knot_derivatives[0][polygon_segments + 1],
                    );
            }
            kv[(derivative, i - (polygon_segments + 1 - degree))] = sum;
        }
    }
    kv
}

fn calculate_kw(curve: &Curve) -> DMatrix<f64> {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();
    let knot_derivatives = &curve.knots.derivatives;
    let point_matrix = curve.control_points.matrix();

    let mut kw = DMatrix::zeros(degree, degree);

    for derivative in 0..=degree - 1 {
        for j in 0..=degree - 1 {
            let mut sum = 0.;

            for basis_index in 0..=degree - derivative {
                sum += prefactor(degree, basis_index, j, derivative, point_matrix, &knot_derivatives[0]) *
                    basis(
                        &knot_derivatives[derivative],
                        basis_index,
                        degree - derivative,
                        0,
                        polygon_segments - derivative,
                        knot_derivatives[0][degree],
                    );
            }
            kw[(derivative, j)] = sum;
        }
    }
    kw *= -1.0;

    kw
}

fn calculate_iv(curve: &Curve) -> DMatrix<f64> {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();
    let knot_derivatives = &curve.knots.derivatives;
    let point_matrix = curve.control_points.matrix();

    let mut iv = DMatrix::zeros(degree, degree);

    for i in polygon_segments - degree + 1..=polygon_segments {
        for derivative in 0..=degree - 1 {
            let mut sum = 0.;

            for basis_index in polygon_segments - degree..=polygon_segments - derivative {
                sum += prefactor(degree, basis_index, i, derivative, point_matrix, &knot_derivatives[0]) *
                    basis(
                        &knot_derivatives[derivative],
                        basis_index,
                        degree - derivative,
                        0,
                        polygon_segments - derivative,
                        knot_derivatives[0][polygon_segments + 1],
                    );
            }
            iv[(i - (polygon_segments + 1 - degree), derivative)] = sum;
        }
    }
    iv *= 0.5;

    iv
}

fn calculate_jw(curve: &Curve) -> DMatrix<f64> {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();
    let knot_derivatives = &curve.knots.derivatives;
    let point_matrix = curve.control_points.matrix();

    let mut jw = DMatrix::zeros(degree, degree);

    for j in 0..=degree - 1 {
        for derivative in 0..=degree - 1 {
            let mut sum = 0.;

            for basis_index in 0..=degree - derivative {
                sum += prefactor(degree, basis_index, j, derivative, point_matrix, &knot_derivatives[0]) *
                    basis(
                        &knot_derivatives[derivative],
                        basis_index,
                        degree - derivative,
                        0,
                        polygon_segments - derivative,
                        knot_derivatives[0][degree],
                    );
            }
            jw[(j, derivative)] = sum;
        }
    }
    jw *= -0.5;

    jw
}

fn calculate_gv(curve: &Curve, parameters: &[f64]) -> DMatrix<f64> {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();
    let knot_values = curve.knots.vector();

    let mut gv = DMatrix::zeros(parameters.len(), degree);

    for (g, &u) in parameters.iter().enumerate() {
        for i in polygon_segments - degree + 1..=polygon_segments {
            gv[(g, i - (polygon_segments - degree + 1))] = basis(knot_values, i, degree, 0, polygon_segments, u);
        }
    }
    gv
}

fn calculate_hw(curve: &Curve, parameters: &[f64]) -> DMatrix<f64> {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();
    let knot_values = curve.knots.vector();

    let mut hw = DMatrix::zeros(parameters.len(), degree);

    for (h, &u) in parameters.iter().enumerate() {
        for i in 0..=degree - 1 {
            hw[(h, i)] = basis(knot_values, i, degree, 0, polygon_segments, u);
        }
    }

    hw
}

fn calculate_ipv(curve: &Curve, parameters: &[f64]) -> DMatrix<f64> {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();
    let knot_values = curve.knots.vector();

    let mut ipv = DMatrix::zeros(degree, parameters.len());

    for i in polygon_segments - degree + 1..=polygon_segments {
        for (g, &u) in parameters.iter().enumerate() {
            ipv[(i - (polygon_segments + 1 - degree), g)] = basis(knot_values, i, degree, 0, polygon_segments, u);
        }
    }
    ipv *= -0.5;
    ipv
}

fn calculate_jppw(curve: &Curve, parameters: &[f64]) -> DMatrix<f64> {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();
    let knot_values = curve.knots.vector();

    let mut jppw = DMatrix::zeros(degree, parameters.len());

    for j in 0..=degree - 1 {
        for (h, &u) in parameters.iter().enumerate() {
            jppw[(j, h)] = basis(knot_values, j, degree, 0, polygon_segments, u);
        }
    }
    jppw *= -0.5;
    jppw
}

fn calculate_kconst(left: &Curve, right: &Curve) -> DMatrix<f64> {
    let degree = left.degree();
    let dimension = left.dimension();

    let mut kconst = DMatrix::zeros(dimension, degree);
    let mut sum = DVector::zeros(dimension);

    let left_polygon_segments = left.polygon_segments();
    let right_polygon_segments = right.polygon_segments();

    let left_knot_derivatives = &left.knots.derivatives;
    let right_knot_derivatives = &right.knots.derivatives;

    let left_points = left.control_points.matrix();
    let right_points = right.control_points.matrix();

    for derivative in 0..=degree - 1 {
        sum.fill(0.0);

        for i in left_polygon_segments - degree..=left_polygon_segments {
            for basis_index in left_polygon_segments - degree..=left_polygon_segments - derivative {
                sum += prefactor(degree, basis_index, i, derivative, left_points, &left_knot_derivatives[0]) *
                    basis(
                        &left_knot_derivatives[derivative],
                        basis_index,
                        degree - derivative,
                        0,
                        left_polygon_segments - derivative,
                        left_knot_derivatives[0][left_polygon_segments + 1],
                    ) *
                    left_points.column(i);
            }
        }

        for j in 0..=degree {
            for basis_index in 0..=degree - derivative {
                sum -= prefactor(degree, basis_index, j, derivative, right_points, &right_knot_derivatives[0]) *
                    basis(
                        &right_knot_derivatives[derivative],
                        basis_index,
                        degree - derivative,
                        0,
                        right_polygon_segments - derivative,
                        right_knot_derivatives[0][degree],
                    ) *
                    right_points.column(j);
            }
        }
        kconst.column_mut(derivative).sub_assign(&sum);
    }
    kconst
}

fn calculate_constant_terms(left: &Curve, right: &Curve, total_constraints: usize) -> DMatrix<f64> {
    let degree = left.degree();
    let dimension = left.dimension();

    let mut constant_terms = DMatrix::zeros(dimension, 3 * degree + total_constraints);

    let kconst = calculate_kconst(left, right);
    constant_terms.view_mut((0, 2 * degree), (dimension, degree)).copy_from(&kconst);

    constant_terms
}

fn solve_linear_equation_system(left: &Curve, right: &Curve, constraints: &Constraints) -> Result<DMatrix<f64>> {
    let system_matrix = calculate_system_matrix(left, right, constraints);
    let constant_terms = calculate_constant_terms(left, right, constraints.count());

    Ok(decompose(system_matrix)?
        .solve(&constant_terms.transpose(), f64::EPSILON.sqrt())
        .expect("the SVD was computed with both U and V^T")
        .transpose())
}

fn shift_boundary_control_points(left: &Curve, right: &Curve, shifts: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
    let degree = left.degree();
    let mut left_shifted = left.control_points.matrix().clone();
    let mut right_shifted = right.control_points.matrix().clone();

    left_shifted.columns_mut(left_shifted.ncols() - degree, degree).add_assign(shifts.columns(0, degree));

    right_shifted.columns_mut(0, degree).add_assign(shifts.columns(degree, degree));

    (left_shifted, right_shifted)
}

fn adjust_knot_vectors(left: &Curve, right: &Curve) -> (DVector<f64>, DVector<f64>, DVector<f64>) {
    let degree = left.degree();

    let left_polygon_segments = left.polygon_segments();
    let right_polygon_segments = right.polygon_segments();

    let left_knots = left.knots.vector();
    let right_knots = right.knots.vector();

    let left_adjusted = adjust_knots(degree, left_knots, left_polygon_segments, right_knots);

    let left_reversed = reversed(left_knots);
    let right_reversed = reversed(right_knots);
    let right_adjusted_reversed = adjust_knots(degree, &right_reversed, right_polygon_segments, &left_reversed);
    let right_adjusted = reversed(&right_adjusted_reversed).add_scalar(1.);

    (left_adjusted, right_reversed, right_adjusted)
}

fn adjust_knots(
    degree: usize,
    knots: &DVector<f64>,
    polygon_segments: usize,
    next_knots: &DVector<f64>,
) -> DVector<f64> {
    let mut adjusted = DVector::zeros(polygon_segments + degree + 2);

    adjusted.head_mut(polygon_segments + 2).copy_from(&knots.head(polygon_segments + 2));

    adjusted.tail_mut(degree).copy_from(&next_knots.segment(degree + 1, degree).add_scalar(1.));

    adjusted
}

fn merge_knot_vectors(
    left: &Curve,
    right: &Curve,
    left_adjusted: &DVector<f64>,
    right_adjusted: &DVector<f64>,
) -> DVector<f64> {
    let left_polygon_segments = left.polygon_segments();
    let right_polygon_segments = right.polygon_segments();

    let mut merged_knots = DVector::zeros(left_polygon_segments + 2 + right_polygon_segments + 1);

    merged_knots.head_mut(left_polygon_segments + 2).copy_from(&left_adjusted.head(left_polygon_segments + 2));
    merged_knots.tail_mut(right_polygon_segments + 1).copy_from(&right_adjusted.tail(right_polygon_segments + 1));

    // The concatenated knot vector spans [0, 2]. Normalize it to [0, 1].
    merged_knots.div_assign(merged_knots[left_polygon_segments + right_polygon_segments + 2]);

    merged_knots
}

fn calculate_derivative_control_point(
    index: usize,
    derivative: usize,
    points: &DMatrix<f64>,
    knot_values: &DVector<f64>,
    degree: usize,
    polygon_segments: usize,
    dimension: usize,
) -> DVector<f64> {
    let mut control_point = DVector::zeros(dimension);

    assert!(
        index <= polygon_segments - derivative,
        "the index {} exceeds the last control point {} of the derivative {}",
        index,
        polygon_segments - derivative,
        derivative
    );
    for zero_order_index in 0..=polygon_segments {
        control_point += prefactor(degree, index, zero_order_index, derivative, points, knot_values) *
            points.column(zero_order_index);
    }
    control_point
}

fn adjust_shifted_control_points(
    points: &DMatrix<f64>,
    knot_values: &DVector<f64>,
    adjusted_knot_values: &DVector<f64>,
    degree: usize,
    polygon_segments: usize,
    dimension: usize,
) -> DMatrix<f64> {
    let mut adjusted = DMatrix::zeros(dimension, polygon_segments + 1);
    let mut derivative_points: Vec<Vec<DVector<f64>>> = vec![Vec::new(); polygon_segments + 1];

    for (i, elem) in derivative_points.iter_mut().enumerate().take(polygon_segments + 1) {
        elem.push(points.column(i).into());
    }

    for derivative in 0..=degree - 1 {
        let derivative_point = calculate_derivative_control_point(
            polygon_segments - degree + 1,
            derivative,
            points,
            knot_values,
            degree,
            polygon_segments,
            dimension,
        );
        derivative_points[polygon_segments - degree + 1].push(derivative_point);
    }

    for i in polygon_segments - degree + 2..=polygon_segments {
        derivative_points[i].resize(polygon_segments - i + 1, DVector::zeros(dimension));
        for derivative in (0..=polygon_segments - i).rev() {
            derivative_points[i][derivative] = ((adjusted_knot_values[i + degree] -
                adjusted_knot_values[i + derivative]) /
                ((degree - derivative) as f64)) *
                &derivative_points[i - 1][derivative + 1] +
                &derivative_points[i - 1][derivative];
        }
    }

    for (i, elem) in derivative_points.iter_mut().enumerate().take(polygon_segments + 1) {
        adjusted.set_column(i, &elem[0]);
    }

    adjusted
}

fn adjust_control_points_of_both_curves(
    left: &Curve,
    right: &Curve,
    left_shifted: &DMatrix<f64>,
    right_shifted: &DMatrix<f64>,
    left_adjusted: &DVector<f64>,
    right_reversed: &DVector<f64>,
    right_adjusted: &DVector<f64>,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let degree = left.degree();
    let dimension = left.dimension();

    let left_points = adjust_shifted_control_points(
        left_shifted,
        left.knots.vector(),
        left_adjusted,
        degree,
        left.polygon_segments(),
        dimension,
    );

    // The right curve is adjusted in its reversed orientation and then reversed back.
    let right_shifted_reversed = points::reversed(right_shifted);
    let right_points_reversed = adjust_shifted_control_points(
        &right_shifted_reversed,
        right_reversed,
        right_adjusted,
        degree,
        right.polygon_segments(),
        dimension,
    );
    let right_points = points::reversed(&right_points_reversed);

    (left_points, right_points)
}

fn merge_control_points(
    left: &Curve,
    right: &Curve,
    left_points: &DMatrix<f64>,
    right_points: &DMatrix<f64>,
) -> DMatrix<f64> {
    let degree = left.degree();
    let dimension = left.dimension();
    let left_count = left.control_points.count();
    let right_count = right.control_points.count();

    let mut merged_points = DMatrix::zeros(dimension, left_count + right_count - degree);

    merged_points.columns_mut(0, left_count).copy_from(left_points);

    let tail_count = right_count + 1 - degree;
    merged_points
        .columns_mut(merged_points.ncols() - tail_count, tail_count)
        .copy_from(&right_points.columns(right_points.ncols() - tail_count, tail_count));

    merged_points
}

fn kronecker_delta(i: usize, j: usize) -> bool {
    i == j
}

/// Returns the factor that ties a control point of the `k`-th derivative curve
/// to a zero-order control point — see `Tai2003`. It evaluates the orders from 0 upward,
/// so the time grows with k² and not with 2ᵏ.
fn prefactor(
    degree: usize,
    index: usize,
    zero_order_index: usize,
    derivative: usize,
    points: &DMatrix<f64>,
    knot_values: &DVector<f64>,
) -> f64 {
    let polygon_segments = points.ncols() - 1;

    with_buffer(derivative + 1, |factors| {
        // The factors of order 0 from the index i to i + k.
        for (offset, factor) in factors.iter_mut().enumerate() {
            let j = index + offset;
            *factor = if j <= polygon_segments && kronecker_delta(j, zero_order_index) { 1. } else { 0. };
        }

        // Each order combines two neighbors of the order below and keeps one factor fewer.
        for order in 1..=derivative {
            for offset in 0..=derivative - order {
                let j = index + offset;
                factors[offset] =
                    if j + order > polygon_segments || knot_values[j + degree + 1] == knot_values[j + order] {
                        0.
                    } else {
                        (degree + 1 - order) as f64 / (knot_values[j + degree + 1] - knot_values[j + order]) *
                            (factors[offset + 1] - factors[offset])
                    };
            }
        }
        factors[0]
    })
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};

    use crate::Curve;

    use super::*;

    fn test_curve(degree: usize, points: DMatrix<f64>) -> Curve {
        Curve::with_uniform_knots(ControlPoints::new(points), degree).unwrap()
    }

    #[test]
    fn merge_of_degree_20_keeps_the_outer_end_points() {
        let degree = 20;
        let count = degree + 2;
        let left = test_curve(degree, DMatrix::from_fn(1, count, |_, column| column as f64));
        let right = test_curve(degree, DMatrix::from_fn(1, count, |_, column| (count + column) as f64));

        let merged = merge(&left, &right, &Constraints::default()).unwrap();

        assert_eq!(merged.evaluate(0.0).unwrap(), left.evaluate(0.0).unwrap());
        assert_eq!(merged.evaluate(1.0).unwrap(), right.evaluate(1.0).unwrap());
    }

    #[test]
    fn merge_accepts_curves_with_different_numbers_of_control_points() {
        let degree = 2;
        let long = test_curve(degree, dmatrix![-6., -5., -4., -3., -2., -1.;]);
        let short = test_curve(degree, dmatrix![1., 2., 3.;]);

        for (left, right) in [(&long, &short), (&short, &long)] {
            let merged = merge(left, right, &Constraints::default()).unwrap();

            let expected_count = left.control_points.count() + right.control_points.count() - degree;
            assert_eq!(merged.control_points.count(), expected_count);
            assert_eq!(merged.evaluate(0.0).unwrap(), left.evaluate(0.0).unwrap());
            assert_eq!(merged.evaluate(1.0).unwrap(), right.evaluate(1.0).unwrap());
        }
    }

    #[test]
    fn merge_errors_for_degree_0() {
        let curve = test_curve(1, dmatrix![0., 1.;]).derivative_curve(1).unwrap();
        assert_eq!(curve.degree(), 0, "the first derivative of a line has degree 0");
        assert_eq!(merge(&curve, &curve, &Constraints::default()).err(), Some(Error::DegreeTooLow { degree: 0 }));
    }

    #[test]
    fn merge_errors_for_a_constraint_outside_the_domain() {
        let left = test_curve(3, dmatrix![-4., -3., -2., -1.;]);
        let right = test_curve(3, dmatrix![1., 2., 3., 4.;]);

        let u = 1.5;
        let constraints = Constraints { left: vec![u], right: vec![] };
        assert_eq!(merge(&left, &right, &constraints).err(), Some(Error::OutsideDomain { u, min: 0.0, max: 1.0 }));

        let constraints = Constraints { left: vec![], right: vec![f64::NAN] };
        assert!(matches!(merge(&left, &right, &constraints), Err(Error::OutsideDomain { .. })));
    }

    mod knots {
        use super::*;

        #[test]
        fn merge_concatenates_the_knot_vectors() {
            let curve = merge(
                &test_curve(1, dmatrix![-2.,-1.,0.;]),
                &test_curve(1, dmatrix![0.,1.,2.;]),
                &Constraints::default(),
            )
            .unwrap();
            assert_eq!(curve.knots.vector(), &dvector![0., 0., 0.25, 0.5, 0.75, 1., 1.]);
        }
    }

    mod control_points {
        use std::ops::Mul;

        use approx::assert_relative_eq;

        use crate::points;

        use super::*;

        #[test]
        fn no_shift_degree_1() {
            let degree = 1;
            let points = dmatrix![
                0.,1.,2.;
                0.,1.,2.;
            ];
            let curve = merge(
                &test_curve(degree, points::reversed(&points).mul(-1.)),
                &test_curve(degree, points),
                &Constraints::default(),
            )
            .unwrap();
            assert_relative_eq!(
                curve.control_points.matrix(),
                &dmatrix![
                    -2.,-1.,0.,1.,2.;
                    -2.,-1.,0.,1.,2.;
                ],
                epsilon = f64::EPSILON.sqrt()
            );
        }

        #[test]
        fn no_shift_degree_2() {
            let degree = 2;
            let points = dmatrix![0.,1.,2.;];
            let curve = merge(
                &test_curve(degree, points::reversed(&points).mul(-1.)),
                &test_curve(degree, points),
                &Constraints::default(),
            )
            .unwrap();
            assert_relative_eq!(
                curve.control_points.matrix(),
                &dmatrix![-2.,-1.,1.,2.;],
                epsilon = f64::EPSILON.sqrt()
            );
        }

        #[test]
        fn shift_degree_1() {
            let degree = 1;
            let points = dmatrix![0.5,1.,2.;];
            let curve = merge(
                &test_curve(degree, points::reversed(&points).mul(-1.)),
                &test_curve(degree, points),
                &Constraints::default(),
            )
            .unwrap();
            assert_relative_eq!(
                curve.control_points.matrix(),
                &dmatrix![-2.,-1.,0.,1.,2.;],
                epsilon = f64::EPSILON.sqrt()
            );
        }

        #[test]
        fn shift_degree_2() {
            let degree = 2;
            let points = dmatrix![0.5,1.,2.;];
            let curve = merge(
                &test_curve(degree, points::reversed(&points).mul(-1.)),
                &test_curve(degree, points),
                &Constraints::default(),
            )
            .unwrap();
            assert_relative_eq!(
                curve.control_points.matrix(),
                &dmatrix![-2.,-1.,1.,2.;],
                epsilon = f64::EPSILON.sqrt()
            );
        }

        #[test]
        fn shift_constrain_left_degree_2() {
            let degree = 2;
            let points = dmatrix![0.5,1.,2.;];
            let curve = merge(
                &test_curve(degree, points::reversed(&points).mul(-1.)),
                &test_curve(degree, points),
                &Constraints { left: vec![1.], right: vec![] },
            )
            .unwrap();

            assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);

            assert_relative_eq!(
                curve.control_points.matrix(),
                &dmatrix![-2.,-1.5,0.5,2.;],
                epsilon = f64::EPSILON.sqrt()
            );
        }

        #[test]
        fn shift_constrain_right_degree_2() {
            let degree = 2;
            let points = dmatrix![0.5,1.,2.;];
            let curve = merge(
                &test_curve(degree, points::reversed(&points).mul(-1.)),
                &test_curve(degree, points),
                &Constraints { left: vec![], right: vec![0.] },
            )
            .unwrap();

            assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);

            assert_relative_eq!(
                curve.control_points.matrix(),
                &dmatrix![-2.,-0.5,1.5,2.;],
                epsilon = f64::EPSILON.sqrt()
            );
        }
    }
}
