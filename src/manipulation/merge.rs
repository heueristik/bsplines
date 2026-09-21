#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("merge-before", "doc-images/plots/manipulation/merge-before.svg"),
doc = ::embed_doc_image::embed_image!("merge-after", "doc-images/plots/manipulation/merge-after.svg"),
doc = ::embed_doc_image::embed_image!("merge-after-left-end-constrained", "doc-images/plots/manipulation/merge-after-left-end-constrained.svg"),
doc = ::embed_doc_image::embed_image!("merge-after-right-start-constrained", "doc-images/plots/manipulation/merge-after-right-start-constrained.svg")))]
//! Combines two independent curves into one.
//!
//! | Two curves before merging.                   | The resulting curve after merging.              |
//! |:--------------------------------------------:|:-----------------------------------------------:|
//! | ![][merge-before]                            | ![][merge-after]                                |
//!
//! Constraints can be set so that certain points on the curve stay fixed.
//!
//! | After merging with the left end constrained. | After merging with the right start constrained. |
//! |:--------------------------------------------:|:-----------------------------------------------:|
//! | ![][merge-after-left-end-constrained]        | ![][merge-after-right-start-constrained]        |

use std::ops::{AddAssign, DivAssign, SubAssign};

use nalgebra::SVD;

use crate::{
    Curve,
    basis::basis,
    error::{Error, Result},
    knots::{Knots, is_clamped, is_normalized, reversed},
    points,
    points::{ControlPoints, Points},
    types::{MatD, VecD, VecHelpers},
};

/// The parameters at which points on a curve stay fixed during a merge.
pub struct Constraints {
    /// The constrained parameter values.
    pub parameters: Vec<f64>,
}

impl Constraints {
    /// Returns the number of constrained points.
    pub fn count(&self) -> usize {
        self.parameters.len()
    }

    /// Returns the number of segments between the constrained points — one less than their count.
    pub fn polyline_segments(&self) -> usize {
        self.count() - 1
    }
}

/// A curve paired with the constraints that hold during a merge.
pub struct ConstrainedCurve<'a> {
    pub(crate) curve: &'a Curve,
    pub(crate) constraints: Constraints,
}

/// Merges two curves, keeping the start of the left curve fixed.
pub fn merge_from(left: &Curve, right: &Curve) -> Result<Curve> {
    merge_with_constraints(
        &ConstrainedCurve { curve: left, constraints: Constraints { parameters: vec![1.] } },
        &ConstrainedCurve { curve: right, constraints: Constraints { parameters: vec![] } },
    )
}

/// Merges two curves, keeping the end of the right curve fixed.
pub fn merge_to(left: &Curve, right: &Curve) -> Result<Curve> {
    merge_with_constraints(
        &ConstrainedCurve { curve: left, constraints: Constraints { parameters: vec![] } },
        &ConstrainedCurve { curve: right, constraints: Constraints { parameters: vec![0.] } },
    )
}

/// Merges two curves into one, attaching the end of the left curve to the start
/// of the right one while maintaining continuity of all derivatives — see `Tai2003`.
pub fn merge(left: &Curve, right: &Curve) -> Result<Curve> {
    merge_with_constraints(
        &ConstrainedCurve { curve: left, constraints: Constraints { parameters: vec![] } },
        &ConstrainedCurve { curve: right, constraints: Constraints { parameters: vec![] } },
    )
}

pub(crate) fn merge_with_constraints(left: &ConstrainedCurve, right: &ConstrainedCurve) -> Result<Curve> {
    let left_degree = left.curve.degree();
    let right_degree = right.curve.degree();

    if left_degree != right_degree {
        return Err(Error::DegreeMismatch { left: left_degree, right: right_degree });
    }

    if left.curve.dimension() != right.curve.dimension() {
        return Err(Error::DimensionMismatch { left: left.curve.dimension(), right: right.curve.dimension() });
    }

    if !is_clamped(&left.curve.knots) || !is_clamped(&right.curve.knots) {
        return Err(Error::UnclampedCurve);
    }

    if !is_normalized(&left.curve.knots) || !is_normalized(&right.curve.knots) {
        return Err(Error::UnnormalizedCurve);
    }

    let total_constraints = left.constraints.count() + right.constraints.count();
    if total_constraints >= left_degree {
        return Err(Error::TooManyConstraints { total: total_constraints, degree: left_degree });
    }

    let shifts = solve_linear_equation_system(left, right);
    let (left_shifted, right_shifted) = shift_boundary_control_points(left.curve, right.curve, &shifts);

    let (left_adjusted, right_reversed, right_adjusted) = adjust_knot_vectors(left.curve, right.curve);
    let merged_knots = merge_knot_vectors(left.curve, right.curve, &left_adjusted, &right_adjusted);

    let (left_points, right_points) = adjust_control_points_of_both_curves(
        left.curve,
        &left_shifted,
        &right_shifted,
        &left_adjusted,
        &right_reversed,
        &right_adjusted,
    );

    let merged_points = merge_control_points(left.curve, right.curve, &left_points, &right_points);

    Curve::new(Knots::new(left_degree, merged_knots), ControlPoints::new(merged_points))
}

// The names of the block matrices (kv, kw, iv, jw, gv, hw, ipv, jppw, kconst) follow the notation in `Tai2003`.
fn construct_system_matrix(left: &ConstrainedCurve, right: &ConstrainedCurve) -> MatD {
    let degree = left.curve.degree();

    let left_constraints = left.constraints.count();
    let right_constraints = right.constraints.count();

    let dimension = 3 * degree + left_constraints + right_constraints;
    let mut system_matrix = MatD::zeros(dimension, dimension);

    system_matrix.view_mut((0, 0), (2 * degree, 2 * degree)).copy_from(&MatD::identity(2 * degree, 2 * degree));

    system_matrix.view_mut((2 * degree, 0), (degree, degree)).copy_from(&calculate_kv(left.curve));
    system_matrix.view_mut((2 * degree, degree), (degree, degree)).copy_from(&calculate_kw(right.curve));

    system_matrix.view_mut((0, 2 * degree), (degree, degree)).copy_from(&calculate_iv(left.curve));
    system_matrix.view_mut((degree, 2 * degree), (degree, degree)).copy_from(&calculate_jw(right.curve));

    if left_constraints > 0 {
        system_matrix.view_mut((3 * degree, 0), (left_constraints, degree)).copy_from(&calculate_gv(left));
        system_matrix.view_mut((0, 3 * degree), (degree, left_constraints)).copy_from(&calculate_ipv(left));
    }
    if right_constraints > 0 {
        system_matrix
            .view_mut((3 * degree + left_constraints, degree), (right_constraints, degree))
            .copy_from(&calculate_hw(right));
        system_matrix
            .view_mut((degree, 3 * degree + left_constraints), (degree, right_constraints))
            .copy_from(&calculate_jppw(right));
    }

    system_matrix
}

fn calculate_kv(curve: &Curve) -> MatD {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();

    let knot_derivatives = &curve.knots.derivatives;
    let point_matrix = curve.points.matrix();

    let mut kv = MatD::zeros(degree, degree);

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

fn calculate_kw(curve: &Curve) -> MatD {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();
    let knot_derivatives = &curve.knots.derivatives;
    let point_matrix = curve.points.matrix();

    let mut kw = MatD::zeros(degree, degree);

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

fn calculate_iv(curve: &Curve) -> MatD {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();
    let knot_derivatives = &curve.knots.derivatives;
    let point_matrix = curve.points.matrix();

    let mut iv = MatD::zeros(degree, degree);

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

fn calculate_jw(curve: &Curve) -> MatD {
    let degree = curve.degree();
    let polygon_segments = curve.polygon_segments();
    let knot_derivatives = &curve.knots.derivatives;
    let point_matrix = curve.points.matrix();

    let mut jw = MatD::zeros(degree, degree);

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

fn calculate_gv(constrained: &ConstrainedCurve) -> MatD {
    let degree = constrained.curve.degree();
    let polygon_segments = constrained.curve.polygon_segments();
    let knot_values = constrained.curve.knots.vector();

    let constraint_segments = constrained.constraints.polyline_segments();

    let mut gv = MatD::zeros(constraint_segments + 1, degree);

    for g in 0..=constraint_segments {
        for i in polygon_segments - degree + 1..=polygon_segments {
            gv[(g, i - (polygon_segments - degree + 1))] =
                basis(knot_values, i, degree, 0, polygon_segments, constrained.constraints.parameters[g]);
        }
    }
    gv
}

fn calculate_hw(constrained: &ConstrainedCurve) -> MatD {
    let degree = constrained.curve.degree();
    let polygon_segments = constrained.curve.polygon_segments();
    let knot_values = constrained.curve.knots.vector();

    let constraint_segments = constrained.constraints.polyline_segments();
    let mut hw = MatD::zeros(constraint_segments + 1, degree);

    for h in 0..=constraint_segments {
        for i in 0..=degree - 1 {
            hw[(h, i)] = basis(knot_values, i, degree, 0, polygon_segments, constrained.constraints.parameters[h]);
        }
    }

    hw
}

fn calculate_ipv(constrained: &ConstrainedCurve) -> MatD {
    let degree = constrained.curve.degree();
    let polygon_segments = constrained.curve.polygon_segments();
    let knot_values = constrained.curve.knots.vector();

    let constraint_segments = constrained.constraints.polyline_segments();

    let mut ipv = MatD::zeros(degree, constraint_segments + 1);

    for i in polygon_segments - degree + 1..=polygon_segments {
        for g in 0..=constraint_segments {
            ipv[(i - (polygon_segments + 1 - degree), g)] =
                basis(knot_values, i, degree, 0, polygon_segments, constrained.constraints.parameters[g]);
        }
    }
    ipv *= -0.5;
    ipv
}

fn calculate_jppw(constrained: &ConstrainedCurve) -> MatD {
    let degree = constrained.curve.degree();
    let polygon_segments = constrained.curve.polygon_segments();
    let knot_values = constrained.curve.knots.vector();

    let constraint_segments = constrained.constraints.polyline_segments();

    let mut jppw = MatD::zeros(degree, constraint_segments + 1);

    for j in 0..=degree - 1 {
        for h in 0..=constraint_segments {
            jppw[(j, h)] = basis(knot_values, j, degree, 0, polygon_segments, constrained.constraints.parameters[h]);
        }
    }
    jppw *= -0.5;
    jppw
}

fn calculate_kconst(left: &Curve, right: &Curve) -> MatD {
    let degree = left.degree();
    let dimension = left.dimension();

    let mut kconst = MatD::zeros(dimension, degree);
    let mut sum = VecD::zeros(dimension);

    let left_polygon_segments = left.polygon_segments();
    let right_polygon_segments = right.polygon_segments();

    let left_knot_derivatives = &left.knots.derivatives;
    let right_knot_derivatives = &right.knots.derivatives;

    let left_points = left.points.matrix();
    let right_points = right.points.matrix();

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

fn construct_constant_terms(left: &Curve, right: &Curve, total_constraints: usize) -> MatD {
    let degree = left.degree();
    let dimension = left.dimension();

    let mut constant_terms = MatD::zeros(dimension, 3 * degree + total_constraints);

    let kconst = calculate_kconst(left, right);
    constant_terms.view_mut((0, 2 * degree), (dimension, degree)).copy_from(&kconst);

    constant_terms
}

fn solve_linear_equation_system(left: &ConstrainedCurve, right: &ConstrainedCurve) -> MatD {
    let total_constraints = left.constraints.count() + right.constraints.count();

    let system_matrix = construct_system_matrix(left, right);
    let constant_terms = construct_constant_terms(left.curve, right.curve, total_constraints);

    SVD::new(system_matrix, true, true)
        .solve(&constant_terms.transpose(), f64::EPSILON.sqrt())
        .expect("the SVD was computed with both U and V^T")
        .transpose()
}

fn shift_boundary_control_points(left: &Curve, right: &Curve, shifts: &MatD) -> (MatD, MatD) {
    let degree = left.degree();
    let mut left_shifted = left.points.matrix().clone();
    let mut right_shifted = right.points.matrix().clone();

    left_shifted.columns_mut(left_shifted.ncols() - degree, degree).add_assign(shifts.columns(0, degree));

    right_shifted.columns_mut(0, degree).add_assign(shifts.columns(degree, degree));

    (left_shifted, right_shifted)
}

fn adjust_knot_vectors(left: &Curve, right: &Curve) -> (VecD, VecD, VecD) {
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

fn adjust_knots(degree: usize, knots: &VecD, polygon_segments: usize, next_knots: &VecD) -> VecD {
    let mut adjusted = VecD::zeros(polygon_segments + degree + 2);

    adjusted.head_mut(polygon_segments + 2).copy_from(&knots.head(polygon_segments + 2));

    adjusted.tail_mut(degree).copy_from(&next_knots.segment(degree + 1, degree).add_scalar(1.));

    adjusted
}

fn merge_knot_vectors(left: &Curve, right: &Curve, left_adjusted: &VecD, right_adjusted: &VecD) -> VecD {
    let left_polygon_segments = left.polygon_segments();
    let right_polygon_segments = right.polygon_segments();

    let mut merged_knots = VecD::zeros(left_polygon_segments + 2 + right_polygon_segments + 1);

    merged_knots.head_mut(left_polygon_segments + 2).copy_from(&left_adjusted.head(left_polygon_segments + 2));
    merged_knots.tail_mut(right_polygon_segments + 1).copy_from(&right_adjusted.tail(right_polygon_segments + 1));

    // The concatenated knot vector spans [0, 2]. Normalize it to [0, 1].
    merged_knots.div_assign(merged_knots[left_polygon_segments + right_polygon_segments + 2]);

    merged_knots
}

fn generate_derivative_control_point(
    index: usize,
    derivative: usize,
    points: &MatD,
    knot_values: &VecD,
    degree: usize,
    polygon_segments: usize,
    dimension: usize,
) -> VecD {
    let mut control_point = VecD::zeros(dimension);

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
    points: &MatD,
    knot_values: &VecD,
    adjusted_knot_values: &VecD,
    degree: usize,
    polygon_segments: usize,
    dimension: usize,
) -> MatD {
    let mut adjusted = MatD::zeros(dimension, polygon_segments + 1);
    let mut derivative_points: Vec<Vec<VecD>> = vec![Vec::new(); polygon_segments + 1];

    for (i, elem) in derivative_points.iter_mut().enumerate().take(polygon_segments + 1) {
        elem.push(points.column(i).into());
    }

    for derivative in 0..=degree - 1 {
        let derivative_point = generate_derivative_control_point(
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
        derivative_points[i].resize(polygon_segments - i + 1, VecD::zeros(dimension));
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
    left_shifted: &MatD,
    right_shifted: &MatD,
    left_adjusted: &VecD,
    right_reversed: &VecD,
    right_adjusted: &VecD,
) -> (MatD, MatD) {
    let degree = left.degree();
    let polygon_segments = left.polygon_segments();
    let dimension = left.dimension();

    let left_knots = left.knots.vector();

    let left_points =
        adjust_shifted_control_points(left_shifted, left_knots, left_adjusted, degree, polygon_segments, dimension);

    // The right curve is adjusted in its reversed orientation and then reversed back.
    let right_shifted_reversed = points::reversed(right_shifted);
    let right_points_reversed = adjust_shifted_control_points(
        &right_shifted_reversed,
        right_reversed,
        right_adjusted,
        degree,
        polygon_segments,
        dimension,
    );
    let right_points = points::reversed(&right_points_reversed);

    (left_points, right_points)
}

fn merge_control_points(left: &Curve, right: &Curve, left_points: &MatD, right_points: &MatD) -> MatD {
    let degree = left.degree();
    let dimension = left.dimension();
    let left_count = left.points.count();
    let right_count = right.points.count();

    let mut merged_points = MatD::zeros(dimension, left_count + right_count - degree);

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
/// to a zero-order control point — see `Tai2003`.
fn prefactor(
    degree: usize,
    index: usize,
    zero_order_index: usize,
    derivative: usize,
    points: &MatD,
    knot_values: &VecD,
) -> f64 {
    let polygon_segments = points.ncols() - 1;

    if index <= polygon_segments - derivative {
        if derivative == 0 {
            if kronecker_delta(index, zero_order_index) { 1. } else { 0. }
        } else if knot_values[index + degree + 1] == knot_values[index + derivative] {
            0.
        } else {
            (degree + 1 - derivative) as f64 / (knot_values[index + degree + 1] - knot_values[index + derivative]) *
                (prefactor(degree, index + 1, zero_order_index, derivative - 1, points, knot_values) -
                    prefactor(degree, index, zero_order_index, derivative - 1, points, knot_values))
        }
    } else {
        0.
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};

    use crate::Curve;

    use super::*;

    fn test_curve(degree: usize, points: MatD) -> Curve {
        Curve::with_uniform_knots(degree, ControlPoints::new(points)).unwrap()
    }

    mod knots {
        use super::*;

        #[test]
        fn merge_concatenates_the_knot_vectors() {
            let curve = merge(&test_curve(1, dmatrix![-2.,-1.,0.;]), &test_curve(1, dmatrix![0.,1.,2.;])).unwrap();
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
            let curve =
                merge(&test_curve(degree, points::reversed(&points).mul(-1.)), &test_curve(degree, points)).unwrap();
            assert_relative_eq!(
                curve.points.matrix(),
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
            let curve =
                merge(&test_curve(degree, points::reversed(&points).mul(-1.)), &test_curve(degree, points)).unwrap();
            assert_relative_eq!(curve.points.matrix(), &dmatrix![-2.,-1.,1.,2.;], epsilon = f64::EPSILON.sqrt());
        }

        #[test]
        fn shift_degree_1() {
            let degree = 1;
            let points = dmatrix![0.5,1.,2.;];
            let curve =
                merge(&test_curve(degree, points::reversed(&points).mul(-1.)), &test_curve(degree, points)).unwrap();
            assert_relative_eq!(curve.points.matrix(), &dmatrix![-2.,-1.,0.,1.,2.;], epsilon = f64::EPSILON.sqrt());
        }

        #[test]
        fn shift_degree_2() {
            let degree = 2;
            let points = dmatrix![0.5,1.,2.;];
            let curve =
                merge(&test_curve(degree, points::reversed(&points).mul(-1.)), &test_curve(degree, points)).unwrap();
            assert_relative_eq!(curve.points.matrix(), &dmatrix![-2.,-1.,1.,2.;], epsilon = f64::EPSILON.sqrt());
        }

        #[test]
        fn shift_constrain_left_degree_2() {
            let degree = 2;
            let points = dmatrix![0.5,1.,2.;];
            let curve =
                merge_from(&test_curve(degree, points::reversed(&points).mul(-1.)), &test_curve(degree, points))
                    .unwrap();

            assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);

            assert_relative_eq!(curve.points.matrix(), &dmatrix![-2.,-1.5,0.5,2.;], epsilon = f64::EPSILON.sqrt());
        }

        #[test]
        fn shift_constrain_right_degree_2() {
            let degree = 2;
            let points = dmatrix![0.5,1.,2.;];
            let curve =
                merge_to(&test_curve(degree, points::reversed(&points).mul(-1.)), &test_curve(degree, points)).unwrap();

            assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);

            assert_relative_eq!(curve.points.matrix(), &dmatrix![-2.,-0.5,1.5,2.;], epsilon = f64::EPSILON.sqrt());
        }
    }
}
