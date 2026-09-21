#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("merge-before", "doc-images/plots/manipulation/merge-before.svg"),
doc = ::embed_doc_image::embed_image!("merge-after", "doc-images/plots/manipulation/merge-after.svg"),
doc = ::embed_doc_image::embed_image!("merge-after-left-end-constrained", "doc-images/plots/manipulation/merge-after-left-end-constrained.svg"),
doc = ::embed_doc_image::embed_image!("merge-after-right-start-constrained", "doc-images/plots/manipulation/merge-after-right-start-constrained.svg")))]
//! Combine two independent curves into one.
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
    pub params: Vec<f64>,
}

impl Constraints {
    /// Returns the number of constrained points.
    pub fn count(&self) -> usize {
        self.params.len()
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
pub fn merge_from(a: &Curve, b: &Curve) -> Result<Curve> {
    merge_with_constraints(
        &ConstrainedCurve { curve: a, constraints: Constraints { params: vec![1.] } },
        &ConstrainedCurve { curve: b, constraints: Constraints { params: vec![] } },
    )
}

/// Merges two curves, keeping the end of the right curve fixed.
pub fn merge_to(a: &Curve, b: &Curve) -> Result<Curve> {
    merge_with_constraints(
        &ConstrainedCurve { curve: a, constraints: Constraints { params: vec![] } },
        &ConstrainedCurve { curve: b, constraints: Constraints { params: vec![0.] } },
    )
}

/// Merges two curves into one, attaching the end of the left curve to the start
/// of the right one while maintaining continuity of all derivatives — see `Tai2003`.
pub fn merge(a: &Curve, b: &Curve) -> Result<Curve> {
    merge_with_constraints(
        &ConstrainedCurve { curve: a, constraints: Constraints { params: vec![] } },
        &ConstrainedCurve { curve: b, constraints: Constraints { params: vec![] } },
    )
}

pub(crate) fn merge_with_constraints(a: &ConstrainedCurve, b: &ConstrainedCurve) -> Result<Curve> {
    let p_a = a.curve.degree();
    let p_b = b.curve.degree();

    if p_a != p_b {
        return Err(Error::DegreeMismatch { left: p_a, right: p_b });
    }

    if a.curve.dimension() != b.curve.dimension() {
        return Err(Error::DimensionMismatch { left: a.curve.dimension(), right: b.curve.dimension() });
    }

    if !is_clamped(&a.curve.knots) || !is_clamped(&b.curve.knots) {
        return Err(Error::UnclampedCurve);
    }

    if !is_normalized(&a.curve.knots) || !is_normalized(&b.curve.knots) {
        return Err(Error::UnnormalizedCurve);
    }

    let total_constraints = a.constraints.count() + b.constraints.count();
    if total_constraints >= p_a {
        return Err(Error::TooManyConstraints { total: total_constraints, degree: p_a });
    }

    let shifts = solve_linear_equation_system(a, b);
    let (s_shifted, t_shifted) = generate_shifted_control_point_vectors_of_spline1and2(a.curve, b.curve, &shifts);

    let (v_adjusted, w_reversed, w_adjusted) = adjust_knots_of_both_splines(a.curve, b.curve);
    let merged_knots = create_merged_knot_vector(a.curve, b.curve, &v_adjusted, &w_adjusted);

    let (s_adjusted, t_adjusted) = adjust_shifted_control_points_of_both_splines(
        a.curve,
        &s_shifted,
        &t_shifted,
        &v_adjusted,
        &w_reversed,
        &w_adjusted,
    );

    let merged_points = generate_control_point_vector_of_merged_spline(a.curve, b.curve, &s_adjusted, &t_adjusted);

    Curve::new(Knots::new(p_a, merged_knots), ControlPoints::new(merged_points))
}

// The names of the block matrices (kv, kw, iv, jw, gv, hw, ipv, jppw) follow the notation in `Tai2003`.
fn construct_n_mat(a: &ConstrainedCurve, b: &ConstrainedCurve) -> MatD {
    let p = a.curve.degree();

    let n_constraints_a = a.constraints.count();
    let n_constraints_b = b.constraints.count();

    let nmat_dim = 3 * p + n_constraints_a + n_constraints_b;
    let mut n_mat = MatD::zeros(nmat_dim, nmat_dim);

    n_mat.view_mut((0, 0), (2 * p, 2 * p)).copy_from(&MatD::identity(2 * p, 2 * p));

    n_mat.view_mut((2 * p, 0), (p, p)).copy_from(&calculate_kv(a.curve));
    n_mat.view_mut((2 * p, p), (p, p)).copy_from(&calculate_kw(b.curve));

    n_mat.view_mut((0, 2 * p), (p, p)).copy_from(&calculate_iv(a.curve));
    n_mat.view_mut((p, 2 * p), (p, p)).copy_from(&calculate_jw(b.curve));

    if n_constraints_a > 0 {
        n_mat.view_mut((3 * p, 0), (n_constraints_a, p)).copy_from(&calculate_gv(a));
        n_mat.view_mut((0, 3 * p), (p, n_constraints_a)).copy_from(&calculate_ipv(a));
    }
    if n_constraints_b > 0 {
        n_mat.view_mut((3 * p + n_constraints_a, p), (n_constraints_b, p)).copy_from(&calculate_hw(b));
        n_mat.view_mut((p, 3 * p + n_constraints_a), (p, n_constraints_b)).copy_from(&calculate_jppw(b));
    }

    n_mat
}

fn calculate_kv(a: &Curve) -> MatD {
    let p = a.degree();
    let m = a.polygon_segments();

    let vk = &a.knots.derivatives;
    let s = a.points.matrix();

    let mut kv = MatD::zeros(p, p);

    for k in 0..=p - 1 {
        for i in m - p + 1..=m {
            let mut sum = 0.;

            for a in m - p..=m - k {
                sum += prefactor(p, a, i, k, s, &vk[0]) * basis(&vk[k], a, p - k, 0, m - k, vk[0][m + 1]);
            }
            kv[(k, i - (m + 1 - p))] = sum;
        }
    }
    kv
}

fn calculate_kw(b: &Curve) -> MatD {
    let p = b.degree();
    let o = b.polygon_segments();
    let wk = &b.knots.derivatives;
    let t = b.points.matrix();

    let mut kw = MatD::zeros(p, p);

    for k in 0..=p - 1 {
        for j in 0..=p - 1 {
            let mut sum = 0.;

            for b in 0..=p - k {
                sum += prefactor(p, b, j, k, t, &wk[0]) * basis(&wk[k], b, p - k, 0, o - k, wk[0][p]);
            }
            kw[(k, j)] = sum;
        }
    }
    kw *= -1.0;

    kw
}

fn calculate_iv(a: &Curve) -> MatD {
    let p = a.degree();
    let m = a.polygon_segments();
    let vk = &a.knots.derivatives;
    let s = a.points.matrix();

    let mut iv = MatD::zeros(p, p);

    for i in m - p + 1..=m {
        for k in 0..=p - 1 {
            let mut sum = 0.;

            for a in m - p..=m - k {
                sum += prefactor(p, a, i, k, s, &vk[0]) * basis(&vk[k], a, p - k, 0, m - k, vk[0][m + 1]);
            }
            iv[(i - (m + 1 - p), k)] = sum;
        }
    }
    iv *= 0.5;

    iv
}

fn calculate_jw(b: &Curve) -> MatD {
    let p = b.degree();
    let o = b.polygon_segments();
    let wk = &b.knots.derivatives;
    let t = b.points.matrix();

    let mut jw = MatD::zeros(p, p);

    for j in 0..=p - 1 {
        for k in 0..=p - 1 {
            let mut sum = 0.;

            for b in 0..=p - k {
                sum += prefactor(p, b, j, k, t, &wk[0]) * basis(&wk[k], b, p - k, 0, o - k, wk[0][p]);
            }
            jw[(j, k)] = sum;
        }
    }
    jw *= -0.5;

    jw
}

fn calculate_gv(a: &ConstrainedCurve) -> MatD {
    let p = a.curve.degree();
    let m = a.curve.polygon_segments();
    let vk0 = a.curve.knots.vector();

    let mg = a.constraints.polyline_segments();

    let mut gv = MatD::zeros(mg + 1, p);

    for g in 0..=mg {
        for i in m - p + 1..=m {
            gv[(g, i - (m - p + 1))] = basis(vk0, i, p, 0, m, a.constraints.params[g]);
        }
    }
    gv
}

fn calculate_hw(b: &ConstrainedCurve) -> MatD {
    let p = b.curve.degree();
    let o = b.curve.polygon_segments();
    let wk0 = b.curve.knots.vector();

    let oh = b.constraints.polyline_segments();
    let mut hw = MatD::zeros(oh + 1, p);

    for h in 0..=oh {
        for i in 0..=p - 1 {
            hw[(h, i)] = basis(wk0, i, p, 0, o, b.constraints.params[h]);
        }
    }

    hw
}

fn calculate_ipv(a: &ConstrainedCurve) -> MatD {
    let p = a.curve.degree();
    let m = a.curve.polygon_segments();
    let vk0 = a.curve.knots.vector();

    let mg = a.constraints.polyline_segments();

    let mut ipv = MatD::zeros(p, mg + 1);

    for i in m - p + 1..=m {
        for g in 0..=mg {
            ipv[(i - (m + 1 - p), g)] = basis(vk0, i, p, 0, m, a.constraints.params[g]);
        }
    }
    ipv *= -0.5;
    ipv
}

fn calculate_jppw(b: &ConstrainedCurve) -> MatD {
    let p_ = b.curve.degree();
    let o_ = b.curve.polygon_segments();
    let wk0 = b.curve.knots.vector();

    let oh_ = b.constraints.polyline_segments();

    let mut jppw = MatD::zeros(p_, oh_ + 1);

    for j in 0..=p_ - 1 {
        for h in 0..=oh_ {
            jppw[(j, h)] = basis(wk0, j, p_, 0, o_, b.constraints.params[h]);
        }
    }
    jppw *= -0.5;
    jppw
}

fn calculate_kconst(a: &Curve, b: &Curve) -> MatD {
    let p = a.degree();
    let dim = a.dimension();

    let mut kconst = MatD::zeros(dim, p);
    let mut sum = VecD::zeros(dim);

    let m = a.polygon_segments();
    let o = b.polygon_segments();

    let vk = &a.knots.derivatives;
    let wk = &b.knots.derivatives;

    let s = a.points.matrix();
    let t = b.points.matrix();

    for k in 0..=p - 1 {
        sum.fill(0.0);

        for i in m - p..=m {
            for a in m - p..=m - k {
                sum += prefactor(p, a, i, k, s, &vk[0]) * basis(&vk[k], a, p - k, 0, m - k, vk[0][m + 1]) * s.column(i);
            }
        }

        for j in 0..=p {
            for b in 0..=p - k {
                sum -= prefactor(p, b, j, k, t, &wk[0]) * basis(&wk[k], b, p - k, 0, o - k, wk[0][p]) * t.column(j);
            }
        }
        kconst.column_mut(k).sub_assign(&sum);
    }
    kconst
}

fn construct_const_mat(a: &Curve, b: &Curve, total_constraints: usize) -> MatD {
    let p = a.degree();
    let dim = a.dimension();

    let mut mat = MatD::zeros(dim, 3 * p + total_constraints);

    let kconst = calculate_kconst(a, b);
    mat.view_mut((0, 2 * p), (dim, p)).copy_from(&kconst);

    mat
}

fn solve_linear_equation_system(a: &ConstrainedCurve, b: &ConstrainedCurve) -> MatD {
    let total_constraints = a.constraints.count() + b.constraints.count();

    let n_mat = construct_n_mat(a, b);
    let const_mat = construct_const_mat(a.curve, b.curve, total_constraints);

    SVD::new(n_mat, true, true)
        .solve(&const_mat.transpose(), f64::EPSILON.sqrt())
        .expect("the SVD was computed with both U and V^T")
        .transpose()
}

fn generate_shifted_control_point_vectors_of_spline1and2(a: &Curve, b: &Curve, shifts: &MatD) -> (MatD, MatD) {
    let p = a.degree();
    let mut s_shifted = a.points.matrix().clone();
    let mut t_shifted = b.points.matrix().clone();

    s_shifted.columns_mut(s_shifted.ncols() - p, p).add_assign(shifts.columns(0, p));

    t_shifted.columns_mut(0, p).add_assign(shifts.columns(p, p));

    (s_shifted, t_shifted)
}

fn adjust_knots_of_both_splines(a: &Curve, b: &Curve) -> (VecD, VecD, VecD) {
    let p = a.degree();

    let m = a.polygon_segments();
    let o = b.polygon_segments();

    let v0 = a.knots.vector();
    let w0 = b.knots.vector();

    let v_adjusted = adjust_knots(p, v0, m, w0);

    let v_reversed = reversed(v0);
    let w_reversed = reversed(w0);
    let w_adjusted_reversed = adjust_knots(p, &w_reversed, o, &v_reversed);
    let w_adjusted = reversed(&w_adjusted_reversed).add_scalar(1.);

    (v_adjusted, w_reversed, w_adjusted)
}

fn adjust_knots(p: usize, u_left: &VecD, n_left: usize, u_right: &VecD) -> VecD {
    let mut u_left_adjusted = VecD::zeros(n_left + p + 2);

    u_left_adjusted.head_mut(n_left + 2).copy_from(&u_left.head(n_left + 2));

    u_left_adjusted.tail_mut(p).copy_from(&u_right.segment(p + 1, p).add_scalar(1.));

    u_left_adjusted
}

fn create_merged_knot_vector(a: &Curve, b: &Curve, v_adjusted: &VecD, w_adjusted: &VecD) -> VecD {
    let m = a.polygon_segments();
    let o = b.polygon_segments();

    let mut merged_knots = VecD::zeros(m + 2 + o + 1);

    merged_knots.head_mut(m + 2).copy_from(&v_adjusted.head(m + 2));
    merged_knots.tail_mut(o + 1).copy_from(&w_adjusted.tail(o + 1));

    // The concatenated knot vector spans [0, 2]. Normalize it to [0, 1].
    merged_knots.div_assign(merged_knots[m + o + 2]);

    merged_knots
}

fn generate_derivative_control_point(
    idx: usize,
    k: usize,
    p_shifted: &MatD,
    u0: &VecD,
    p: usize,
    n: usize,
    dim: usize,
) -> VecD {
    let mut control_point = VecD::zeros(dim);

    assert!(idx <= n - k, "Index out of bounds: only n-k control points exist");
    for zero_order_idx in 0..=n {
        control_point += prefactor(p, idx, zero_order_idx, k, p_shifted, u0) * p_shifted.column(zero_order_idx);
    }
    control_point
}

fn adjust_shifted_control_points(
    p_shifted: &MatD,
    u0: &VecD,
    u_adjusted: &VecD,
    p: usize,
    n: usize,
    dim: usize,
) -> MatD {
    let mut p_adjusted = MatD::zeros(dim, n + 1);
    let mut qki: Vec<Vec<VecD>> = vec![Vec::new(); n + 1];

    for (i, elem) in qki.iter_mut().enumerate().take(n + 1) {
        elem.push(p_shifted.column(i).into());
    }

    for k in 0..=p - 1 {
        let derivative_point = generate_derivative_control_point(n - p + 1, k, p_shifted, u0, p, n, dim);
        qki[n - p + 1].push(derivative_point);
    }

    for i in n - p + 2..=n {
        qki[i].resize(n - i + 1, VecD::zeros(dim));
        for k in (0..=n - i).rev() {
            qki[i][k] =
                ((u_adjusted[i + p] - u_adjusted[i + k]) / ((p - k) as f64)) * &qki[i - 1][k + 1] + &qki[i - 1][k];
        }
    }

    for (i, elem) in qki.iter_mut().enumerate().take(n + 1) {
        p_adjusted.set_column(i, &elem[0]);
    }

    p_adjusted
}

fn adjust_shifted_control_points_of_both_splines(
    a: &Curve,
    s_shifted: &MatD,
    t_shifted: &MatD,
    v_adjusted: &VecD,
    w_reversed: &VecD,
    w_adjusted_reversed: &VecD,
) -> (MatD, MatD) {
    let p = a.degree();
    let n = a.polygon_segments();
    let dim = a.dimension();

    let v0 = a.knots.vector();

    let s_adjusted = adjust_shifted_control_points(s_shifted, v0, v_adjusted, p, n, dim);

    // The right curve is adjusted in its reversed orientation and then reversed back.
    let t_shifted_reversed = points::reversed(t_shifted);
    let t_adjusted_reversed =
        adjust_shifted_control_points(&t_shifted_reversed, w_reversed, w_adjusted_reversed, p, n, dim);
    let t_adjusted = points::reversed(&t_adjusted_reversed);

    (s_adjusted, t_adjusted)
}

fn generate_control_point_vector_of_merged_spline(a: &Curve, b: &Curve, s_adjusted: &MatD, t_adjusted: &MatD) -> MatD {
    let p = a.degree();
    let dim = a.dimension();
    let n_points1 = a.points.count();
    let n_points2 = b.points.count();

    let mut merged_points = MatD::zeros(dim, n_points1 + n_points2 - p);

    merged_points.columns_mut(0, n_points1).copy_from(s_adjusted);

    let right_cols = n_points2 + 1 - p;
    merged_points
        .columns_mut(merged_points.ncols() - right_cols, right_cols)
        .copy_from(&t_adjusted.columns(t_adjusted.ncols() - right_cols, right_cols));

    merged_points
}

fn kronecker_delta(i: usize, j: usize) -> bool {
    i == j
}

/// Returns the factor that ties control point `i` of the `k`-th derivative curve
/// to the zero-order control point `i0` — see `Tai2003`.
fn prefactor(p: usize, i: usize, i0: usize, k: usize, p0: &MatD, u0: &VecD) -> f64 {
    let n = p0.ncols() - 1;

    if i <= n - k {
        if k == 0 {
            if kronecker_delta(i, i0) { 1. } else { 0. }
        } else if u0[i + p + 1] == u0[i + k] {
            0.
        } else {
            (p + 1 - k) as f64 / (u0[i + p + 1] - u0[i + k]) *
                (prefactor(p, i + 1, i0, k - 1, p0, u0) - prefactor(p, i, i0, k - 1, p0, u0))
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

    fn test_bspline(degree: usize, points: MatD) -> Curve {
        Curve::with_uniform_knots(degree, ControlPoints::new(points)).unwrap()
    }

    mod knots {
        use super::*;

        #[test]
        fn knots() {
            let c = merge(&test_bspline(1, dmatrix![-2.,-1.,0.;]), &test_bspline(1, dmatrix![0.,1.,2.;])).unwrap();
            assert_eq!(c.knots.vector(), &dvector![0., 0., 0.25, 0.5, 0.75, 1., 1.]);
        }
    }

    mod control_points {
        use std::ops::Mul;

        use approx::assert_relative_eq;

        use crate::points;

        use super::*;

        #[test]
        fn no_shift_degree_1() {
            let p = 1;
            let mat = dmatrix![
                0.,1.,2.;
                0.,1.,2.;
            ];
            let c = merge(&test_bspline(p, points::reversed(&mat).mul(-1.)), &test_bspline(p, mat)).unwrap();
            assert_relative_eq!(
                c.points.matrix(),
                &dmatrix![
                    -2.,-1.,0.,1.,2.;
                    -2.,-1.,0.,1.,2.;
                ],
                epsilon = f64::EPSILON.sqrt()
            );
        }

        #[test]
        fn no_shift_degree_2() {
            let p = 2;
            let mat = dmatrix![0.,1.,2.;];
            let c = merge(&test_bspline(p, points::reversed(&mat).mul(-1.)), &test_bspline(p, mat)).unwrap();
            assert_relative_eq!(c.points.matrix(), &dmatrix![-2.,-1.,1.,2.;], epsilon = f64::EPSILON.sqrt());
        }

        #[test]
        fn shift_degree_1() {
            let p = 1;
            let mat = dmatrix![0.5,1.,2.;];
            let c = merge(&test_bspline(p, points::reversed(&mat).mul(-1.)), &test_bspline(p, mat)).unwrap();
            assert_relative_eq!(c.points.matrix(), &dmatrix![-2.,-1.,0.,1.,2.;], epsilon = f64::EPSILON.sqrt());
        }

        #[test]
        fn shift_degree_2() {
            let p = 2;
            let mat = dmatrix![0.5,1.,2.;];
            let c = merge(&test_bspline(p, points::reversed(&mat).mul(-1.)), &test_bspline(p, mat)).unwrap();
            assert_relative_eq!(c.points.matrix(), &dmatrix![-2.,-1.,1.,2.;], epsilon = f64::EPSILON.sqrt());
        }

        #[test]
        fn shift_constrain_left_degree_2() {
            let p = 2;
            let mat = dmatrix![0.5,1.,2.;];
            let c = merge_from(&test_bspline(p, points::reversed(&mat).mul(-1.)), &test_bspline(p, mat)).unwrap();

            assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);

            assert_relative_eq!(c.points.matrix(), &dmatrix![-2.,-1.5,0.5,2.;], epsilon = f64::EPSILON.sqrt());
        }

        #[test]
        fn shift_constrain_right_degree_2() {
            let p = 2;
            let mat = dmatrix![0.5,1.,2.;];
            let c = merge_to(&test_bspline(p, points::reversed(&mat).mul(-1.)), &test_bspline(p, mat)).unwrap();

            assert_eq!(c.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);

            assert_relative_eq!(c.points.matrix(), &dmatrix![-2.,-0.5,1.5,2.;], epsilon = f64::EPSILON.sqrt());
        }
    }
}
