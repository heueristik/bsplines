//! Merges two curves into one — see `Tai2003`.

use std::ops::AddAssign;

use nalgebra::{DMatrix, DVector};

use crate::{
    Curve,
    basis::basis,
    buffer::with_buffer,
    error::{Error, Result},
    knots::{Knots, normalize},
    points::{ControlPoints, Points},
    svd::decompose,
    vector_views::VectorViews,
};

/// The parameters at which the points of two merged curves stay fixed.
///
/// The fields name the side of the joint, for [`Curve::append_constrained`] and
/// [`Curve::prepend_constrained`] alike. Both curves together take fewer than p constraints, each in [0, 1].
/// The constraints must not contradict each other: two curves that do not meet cannot keep both joined ends.
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

    let left_adjusted = adjust_knots(left, right);
    let left_points = adjust_shifted_control_points(
        &left_shifted,
        left.knots.vector(),
        &left_adjusted,
        left_degree,
        left.polygon_segments(),
        left.dimension(),
    );

    let merged_knots = merge_knot_vectors(left, right);
    let merged_points = merge_control_points(left, right, &left_points, &right_shifted);

    let merged = Curve::new(Knots::new(left_degree, merged_knots)?, ControlPoints::new(merged_points))?;
    check_constraints(left, right, constraints, &merged)?;
    Ok(merged)
}

/// Checks that the merged curve keeps every constrained point. The merged curve runs through the left curve
/// on [0, 1/2] and through the right curve on [1/2, 1]. Contradicting constraints leave the linear system
/// without a solution, and the least-squares result keeps none of them.
fn check_constraints(left: &Curve, right: &Curve, constraints: &Constraints, merged: &Curve) -> Result<()> {
    let scale = 1.0 + left.control_points.matrix().amax().max(right.control_points.matrix().amax());
    let tolerance = f64::EPSILON.sqrt() * scale;

    let left_parameters = constraints.left.iter().map(|&u| (left, u, u / 2.0));
    let right_parameters = constraints.right.iter().map(|&u| (right, u, (1.0 + u) / 2.0));
    for (curve, u, merged_u) in left_parameters.chain(right_parameters) {
        if (merged.evaluate(merged_u)? - curve.evaluate(u)?).amax() > tolerance {
            return Err(Error::ConflictingConstraints);
        }
    }
    Ok(())
}

// The names of the block matrices (kv, kw, iv, jw, gv, hw, ipv, jppw, kconst) follow the notation in `Tai2003`.
fn calculate_system_matrix(left: &Curve, right: &Curve, constraints: &Constraints) -> DMatrix<f64> {
    let degree = left.degree();

    let left_constraints = constraints.left.len();
    let right_constraints = constraints.right.len();

    let dimension = 3 * degree + left_constraints + right_constraints;
    let mut system_matrix = DMatrix::zeros(dimension, dimension);

    system_matrix.view_mut((0, 0), (2 * degree, 2 * degree)).copy_from(&DMatrix::identity(2 * degree, 2 * degree));

    // Each upper block is the transpose of a lower block, scaled by 1/2 or −1/2.
    let kv = calculate_kv(left);
    let kw = calculate_kw(right);
    let iv = kv.transpose() * 0.5;
    let jw = kw.transpose() * 0.5;
    system_matrix.view_mut((2 * degree, 0), (degree, degree)).copy_from(&kv);
    system_matrix.view_mut((2 * degree, degree), (degree, degree)).copy_from(&kw);
    system_matrix.view_mut((0, 2 * degree), (degree, degree)).copy_from(&iv);
    system_matrix.view_mut((degree, 2 * degree), (degree, degree)).copy_from(&jw);

    if left_constraints > 0 {
        let gv = calculate_gv(left, &constraints.left);
        let ipv = gv.transpose() * -0.5;
        system_matrix.view_mut((3 * degree, 0), (left_constraints, degree)).copy_from(&gv);
        system_matrix.view_mut((0, 3 * degree), (degree, left_constraints)).copy_from(&ipv);
    }
    if right_constraints > 0 {
        let hw = calculate_hw(right, &constraints.right);
        let jppw = hw.transpose() * -0.5;
        system_matrix.view_mut((3 * degree + left_constraints, degree), (right_constraints, degree)).copy_from(&hw);
        system_matrix.view_mut((degree, 3 * degree + left_constraints), (degree, right_constraints)).copy_from(&jppw);
    }

    system_matrix
}

fn calculate_kv(curve: &Curve) -> DMatrix<f64> {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();
    let point_matrix = curve.control_points.matrix();
    let knot_values = curve.knots.vector();

    // At u = 1, the last basis function of each derivative curve is 1 and all others are 0.
    DMatrix::from_fn(degree, degree, |derivative, column| {
        let i = polygon_segments + 1 - degree + column;
        prefactor(degree, polygon_segments - derivative, i, derivative, point_matrix, knot_values)
    })
}

fn calculate_kw(curve: &Curve) -> DMatrix<f64> {
    let degree = curve.degree();
    let point_matrix = curve.control_points.matrix();
    let knot_values = curve.knots.vector();

    // At u = 0, the first basis function of each derivative curve is 1 and all others are 0.
    DMatrix::from_fn(degree, degree, |derivative, j| -prefactor(degree, 0, j, derivative, point_matrix, knot_values))
}

fn calculate_gv(curve: &Curve, parameters: &[f64]) -> DMatrix<f64> {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();
    let knot_values = curve.knots.vector();

    let mut gv = DMatrix::zeros(parameters.len(), degree);

    for (g, &u) in parameters.iter().enumerate() {
        for i in polygon_segments - degree + 1..=polygon_segments {
            gv[(g, i - (polygon_segments - degree + 1))] = basis(knot_values, i, degree, u);
        }
    }
    gv
}

fn calculate_hw(curve: &Curve, parameters: &[f64]) -> DMatrix<f64> {
    let degree = curve.degree();
    let knot_values = curve.knots.vector();

    let mut hw = DMatrix::zeros(parameters.len(), degree);

    for (h, &u) in parameters.iter().enumerate() {
        for i in 0..=degree - 1 {
            hw[(h, i)] = basis(knot_values, i, degree, u);
        }
    }

    hw
}

fn calculate_kconst(left: &Curve, right: &Curve) -> DMatrix<f64> {
    // At its ends, each derivative of a clamped curve equals the first or the last control point of that derivative.
    DMatrix::from_fn(left.dimension(), left.degree(), |row, derivative| {
        let left_points = left.control_points.matrix_derivative(derivative);
        let right_points = right.control_points.matrix_derivative(derivative);
        right_points[(row, 0)] - left_points[(row, left_points.ncols() - 1)]
    })
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

/// Returns the knot vector of the left curve with its last p knots replaced by the first p internal knots
/// of the right curve, moved behind the joint at 1.
fn adjust_knots(left: &Curve, right: &Curve) -> DVector<f64> {
    let degree = left.degree();
    let polygon_segments = left.polygon_segments();

    let mut adjusted = DVector::zeros(polygon_segments + degree + 2);
    adjusted.head_mut(polygon_segments + 2).copy_from(&left.knots.vector().head(polygon_segments + 2));
    adjusted.tail_mut(degree).copy_from(&right.knots.vector().segment(degree + 1, degree).add_scalar(1.));
    adjusted
}

fn merge_knot_vectors(left: &Curve, right: &Curve) -> DVector<f64> {
    let left_polygon_segments = left.polygon_segments();
    let right_polygon_segments = right.polygon_segments();

    let mut merged_knots = DVector::zeros(left_polygon_segments + 2 + right_polygon_segments + 1);

    merged_knots.head_mut(left_polygon_segments + 2).copy_from(&left.knots.vector().head(left_polygon_segments + 2));
    merged_knots
        .tail_mut(right_polygon_segments + 1)
        .copy_from(&right.knots.vector().tail(right_polygon_segments + 1).add_scalar(1.));

    // The concatenated knot vector spans [0, 2].
    normalize(&mut merged_knots);

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

    for derivative in 1..=degree - 1 {
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

/// Returns the control points of the merged curve: the adjusted left points without the last one, followed by the
/// shifted right points from the index p − 1 on. The merged curve has p control points fewer than both curves.
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
    fn merge_errors_for_fixed_ends_that_do_not_meet() {
        let degree = 3;
        let left = test_curve(degree, dmatrix![0., 1., 2., 3.;]);
        let apart = test_curve(degree, dmatrix![10., 11., 12., 13.;]);
        let touching = test_curve(degree, dmatrix![3., 4., 5., 6.;]);
        let both_ends = Constraints { left: vec![1.0], right: vec![0.0] };

        assert_ne!(left.evaluate(1.0).unwrap(), apart.evaluate(0.0).unwrap(), "the curves do not meet");
        assert_eq!(merge(&left, &apart, &both_ends).err(), Some(Error::ConflictingConstraints));

        assert_eq!(left.evaluate(1.0).unwrap(), touching.evaluate(0.0).unwrap(), "the curves meet");
        let merged = merge(&left, &touching, &both_ends).unwrap();
        approx::assert_relative_eq!(merged.evaluate(0.5).unwrap(), left.evaluate(1.0).unwrap(), epsilon = 1e-9);
    }

    #[test]
    fn merge_of_the_halves_of_a_split_curve_restores_the_curve() {
        for degree in 1..=5 {
            let points =
                DMatrix::from_fn(
                    2,
                    2 * degree + 2,
                    |row, column| {
                        if row == 0 { column as f64 } else { (column as f64 * 1.1).sin() }
                    },
                );
            let curve = test_curve(degree, points);
            let (left, right) = curve.split(0.5).unwrap();

            let merged = merge(&left, &right, &Constraints::default()).unwrap();

            for u in (0..=20).map(|step| f64::from(step) / 20.0) {
                approx::assert_relative_eq!(merged.evaluate(u).unwrap(), curve.evaluate(u).unwrap(), epsilon = 1e-9);
            }
        }
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
