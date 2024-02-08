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
use thiserror::Error;

use crate::curve::basis::basis;
use crate::{
    curve,
    curve::{
        knots::{is_clamped, is_normed, reversed, Knots},
        points::{ControlPoints, Points},
        Curve, CurveError,
    },
    manipulation::merge::MergeError::CurveGenerationFailure,
    types::{MatD, VecD, VecHelpers},
};

pub struct Constraints {
    pub params: Vec<f64>,
}

impl Constraints {
    pub fn count(&self) -> usize {
        self.params.len()
    }

    pub fn segments(&self) -> usize {
        self.count() - 1
    }
}

pub struct ConstrainedCurve<'a> {
    pub(crate) curve: &'a Curve,
    pub(crate) constraints: Constraints,
}

#[derive(Error, Debug, PartialEq)]
pub enum MergeError {
    #[error("The degree of the left curve `p = {left}` differs from the right curve `p = {right}`.")]
    DegreeMismatch { left: usize, right: usize },

    #[error("The dimension of the left curve `dim = {left}` differs from the right curve `dim = {right}`.")]
    DimensionMismatch { left: usize, right: usize },

    #[error("Curves must be clamped")]
    UnclampedCurve,

    #[error("Curves must be normed.")]
    UnnormedCurve,

    #[error(
        "The total number of constrained points `{totalConstraints}` must be \
        less than the curve degree `p = {degree}`. \
        Otherwise no solution for the linear system of equations exists."
    )]
    TooManyConstraints { totalConstraints: usize, degree: usize },

    #[error("Curve generation failed with error {err}.")]
    CurveGenerationFailure { err: CurveError },
}

// Keeps the start of spline 1 fixed
pub fn merge_from(a: &Curve, b: &Curve) -> Result<Curve, MergeError> {
    merge_with_constraints(
        &ConstrainedCurve { curve: a, constraints: Constraints { params: vec![1.] } },
        &ConstrainedCurve { curve: b, constraints: Constraints { params: vec![] } },
    )
}

// Keeps the end of spline 2 fixed
pub fn merge_to(a: &Curve, b: &Curve) -> Result<Curve, MergeError> {
    merge_with_constraints(
        &ConstrainedCurve { curve: a, constraints: Constraints { params: vec![] } },
        &ConstrainedCurve { curve: b, constraints: Constraints { params: vec![0.] } },
    )
}

pub fn merge(a: &Curve, b: &Curve) -> Result<Curve, MergeError> {
    merge_with_constraints(
        &ConstrainedCurve { curve: a, constraints: Constraints { params: vec![] } },
        &ConstrainedCurve { curve: b, constraints: Constraints { params: vec![] } },
    )
}

pub(crate) fn merge_with_constraints(a: &ConstrainedCurve, b: &ConstrainedCurve) -> Result<Curve, MergeError> {
    let p_a = a.curve.degree();
    let p_b = b.curve.degree();

    if p_a != p_b {
        return Err(MergeError::DegreeMismatch { left: p_a, right: p_b });
    }

    if a.curve.dimension() != b.curve.dimension() {
        return Err(MergeError::DimensionMismatch { left: a.curve.dimension(), right: b.curve.dimension() });
    }

    if !is_clamped(&a.curve.knots) || !is_clamped(&b.curve.knots) {
        return Err(MergeError::UnclampedCurve);
    }

    if !is_normed(&a.curve.knots) || !is_normed(&b.curve.knots) {
        return Err(MergeError::UnnormedCurve);
    }

    let totalConstraints = a.constraints.count() + b.constraints.count();
    if totalConstraints >= p_a {
        return Err(MergeError::TooManyConstraints { totalConstraints, degree: p_a });
    }

    let shifts = solve_linear_equation_system(a, b);
    let (Sshifted, Tshifted) = generate_shifted_control_point_vectors_of_spline1and2(a.curve, b.curve, &shifts);

    // adjust and merge knot vectors
    let (Vadj, Wrev, Wadj) = adjust_knots_of_both_splines(a.curve, b.curve);
    let U_merged = create_merged_knot_vector(a.curve, b.curve, &Vadj, &Wadj);

    // adjust control points
    let (Sadj, Tadj) =
        adjust_shifted_control_points_of_both_splines(a.curve, &Sshifted, &Tshifted, &Vadj, &Wrev, &Wadj);

    // do actual merge
    let P_merged = generate_control_point_vector_of_merged_spline(a.curve, b.curve, &Sadj, &Tadj);

    Curve::new(Knots::new(p_a, U_merged), ControlPoints::new(P_merged)).map_err(|err| CurveGenerationFailure { err })
}

fn construct_Nmat(a: &ConstrainedCurve, b: &ConstrainedCurve) -> MatD {
    let p = a.curve.degree();

    let n_constraints_a = a.constraints.count();
    let n_constraints_b = b.constraints.count();

    let nmat_dim = 3 * p + n_constraints_a + n_constraints_b;
    let mut Nmat = MatD::zeros(nmat_dim, nmat_dim);

    Nmat.view_mut((0, 0), (2 * p, 2 * p)).copy_from(&MatD::identity(2 * p, 2 * p));

    Nmat.view_mut((2 * p, 0), (p, p)).copy_from(&calculateKV(a.curve));
    Nmat.view_mut((2 * p, p), (p, p)).copy_from(&calculateKW(b.curve));

    Nmat.view_mut((0, 2 * p), (p, p)).copy_from(&calculateIV(a.curve));
    Nmat.view_mut((p, 2 * p), (p, p)).copy_from(&calculateJW(b.curve));

    // calculate block matrices for the constraints
    if n_constraints_a > 0 {
        Nmat.view_mut((3 * p, 0), (n_constraints_a, p)).copy_from(&calculateGV(a));
        Nmat.view_mut((0, 3 * p), (p, n_constraints_a)).copy_from(&calculateIpV(a));
    }
    if n_constraints_b > 0 {
        Nmat.view_mut((3 * p + n_constraints_a, p), (n_constraints_b, p)).copy_from(&calculateHW(b));
        Nmat.view_mut((p, 3 * p + n_constraints_a), (p, n_constraints_b)).copy_from(&calculateJppW(b));
    }

    Nmat
}

fn calculateKV(a: &Curve) -> MatD {
    let p = a.degree();
    let m = a.segments();

    let Vk = &a.knots.Uk;
    let S = a.points.matrix();

    let mut KV = MatD::zeros(p, p);

    for k in 0..=p - 1 {
        for i in m - p + 1..=m {
            let mut sum = 0.;

            for a in m - p..=m - k {
                sum += prefactor(p, a, i, k, S, &Vk[0]) * basis(&Vk[k], a, p - k, 0, m - k, Vk[0][m + 1]);
            }
            KV[(k, i - (m + 1 - p))] = sum;
        }
    }
    KV
}

fn calculateKW(b: &Curve) -> MatD {
    let p = b.degree();
    let o = b.segments();
    let Wk = &b.knots.Uk;
    let T = b.points.matrix();

    let mut KW = MatD::zeros(p, p);

    for k in 0..=p - 1 {
        for j in 0..=p - 1 {
            let mut sum = 0.;

            for b in 0..=p - k {
                sum += prefactor(p, b, j, k, T, &Wk[0]) * basis(&Wk[k], b, p - k, 0, o - k, Wk[0][p]);
            }
            KW[(k, j)] = sum;
        }
    }
    KW *= -1.0;

    KW
}

fn calculateIV(a: &Curve) -> MatD {
    let p = a.degree();
    let m = a.segments();
    let Vk = &a.knots.Uk;
    let S = a.points.matrix();

    let mut IV = MatD::zeros(p, p);

    for i in m - p + 1..=m {
        for k in 0..=p - 1 {
            let mut sum = 0.;

            for a in m - p..=m - k {
                sum += prefactor(p, a, i, k, S, &Vk[0]) * basis(&Vk[k], a, p - k, 0, m - k, Vk[0][m + 1]);
            }
            IV[(i - (m + 1 - p), k)] = sum;
        }
    }
    IV *= 0.5;

    IV
}

fn calculateJW(b: &Curve) -> MatD {
    let p = b.degree();
    let o = b.segments();
    let Wk = &b.knots.Uk;
    let T = b.points.matrix();

    let mut JW = MatD::zeros(p, p);

    for j in 0..=p - 1 {
        for k in 0..=p - 1 {
            let mut sum = 0.;

            for b in 0..=p - k {
                sum += prefactor(p, b, j, k, T, &Wk[0]) * basis(&Wk[k], b, p - k, 0, o - k, Wk[0][p]);
            }
            JW[(j, k)] = sum;
        }
    }
    JW *= -0.5;

    JW
}

fn calculateGV(a: &ConstrainedCurve) -> MatD {
    let p = a.curve.degree();
    let m = a.curve.segments();
    let Vk0 = a.curve.knots.vector();

    let mg = a.constraints.segments();

    let mut GV = MatD::zeros(mg + 1, p);

    for g in 0..=mg {
        for i in m - p + 1..=m {
            GV[(g, i - (m - p + 1))] = basis(Vk0, i, p, 0, m, a.constraints.params[g]);
        }
    }
    GV
}

fn calculateHW(b: &ConstrainedCurve) -> MatD {
    let p = b.curve.degree();
    let o = b.curve.segments();
    let Wk0 = b.curve.knots.vector();

    let oh = b.constraints.segments();
    let mut HW = MatD::zeros(oh + 1, p);

    for h in 0..=oh {
        for i in 0..=p - 1 {
            HW[(h, i)] = basis(Wk0, i, p, 0, o, b.constraints.params[h]);
        }
    }

    // TODO use below
    /*let mut HW = MatD::zeros(constraints_b.count(), p);

    for (h, u) in constraints_b.params.iter().enumerate() {
        for i in 0..=p - 1 {
            HW[(h, i)] = evaluate(i, p, o, Wk0, *u);
        }
    }*/
    HW
}

fn calculateIpV(a: &ConstrainedCurve) -> MatD {
    let p = a.curve.degree();
    let m = a.curve.segments();
    let Vk0 = a.curve.knots.vector();

    let mg = a.constraints.segments();

    let mut IpV = MatD::zeros(p, mg + 1);

    for i in m - p + 1..=m {
        for g in 0..=mg {
            IpV[(i - (m + 1 - p), g)] = basis(Vk0, i, p, 0, m, a.constraints.params[g]);
        }
    }
    IpV *= -0.5;
    IpV
}

fn calculateJppW(b: &ConstrainedCurve) -> MatD {
    let p_ = b.curve.degree();
    let o_ = b.curve.segments();
    let Wk0 = b.curve.knots.vector();

    let oh_ = b.constraints.segments();

    let mut JppW = MatD::zeros(p_, oh_ + 1);

    for j in 0..=p_ - 1 {
        for h in 0..=oh_ {
            JppW[(j, h)] = basis(Wk0, j, p_, 0, o_, b.constraints.params[h]);
        }
    }
    JppW *= -0.5;
    JppW
}

fn calculateKconst(a: &Curve, b: &Curve) -> MatD {
    let p = a.degree();
    let dim = a.dimension();

    let mut Kconst = MatD::zeros(dim, p);
    let mut sum = VecD::zeros(dim);

    let m = a.segments();
    let o = b.segments();

    let Vk = &a.knots.Uk;
    let Wk = &b.knots.Uk;

    let S = a.points.matrix();
    let T = b.points.matrix();

    for k in 0..=p - 1 {
        sum.fill(0.0);

        for i in m - p..=m {
            for a in m - p..=m - k {
                // TODO use point getter instead of .column(i)
                sum += prefactor(p, a, i, k, S, &Vk[0]) * basis(&Vk[k], a, p - k, 0, m - k, Vk[0][m + 1]) * S.column(i);
            }
        }

        for j in 0..=p {
            for b in 0..=p - k {
                //sumB += -prefactor(p, b, j, k, &T, &Wk[0]) * evaluate(b, p - k, o - k, &Wk[k], Wk[0][p]) * T.row(j);
                // TODO use point getter instead of .column(j)
                sum -= prefactor(p, b, j, k, T, &Wk[0]) * basis(&Wk[k], b, p - k, 0, o - k, Wk[0][p]) * T.column(j);
            }
        }
        Kconst.column_mut(k).sub_assign(&sum);
    }
    Kconst
}

fn constructConstMat(a: &Curve, b: &Curve, total_constraints: usize) -> MatD {
    let p = a.degree();
    let dim = a.dimension();

    let mut mat = MatD::zeros(dim, 3 * p + total_constraints);

    let Kconst = calculateKconst(a, b);
    mat.view_mut((0, 2 * p), (dim, p)).copy_from(&Kconst);

    mat
}

fn solve_linear_equation_system(a: &ConstrainedCurve, b: &ConstrainedCurve) -> MatD {
    let total_constraints = a.constraints.count() + b.constraints.count();

    let Nmat = construct_Nmat(a, b);
    let const_mat = constructConstMat(a.curve, b.curve, total_constraints);

    SVD::new(Nmat, true, true).solve(&const_mat.transpose(), f64::EPSILON.sqrt()).unwrap().transpose()
}

fn generate_shifted_control_point_vectors_of_spline1and2(a: &Curve, b: &Curve, shifts: &MatD) -> (MatD, MatD) {
    let p = a.degree();
    let mut Sshifted = a.points.matrix().clone();
    let mut Tshifted = b.points.matrix().clone();

    Sshifted.columns_mut(Sshifted.ncols() - p, p).add_assign(shifts.columns(0, p));

    Tshifted.columns_mut(0, p).add_assign(shifts.columns(p, p));

    (Sshifted, Tshifted)
}

fn adjust_knots_of_both_splines(a: &Curve, b: &Curve) -> (VecD, VecD, VecD) {
    let p = a.degree();

    let m = a.segments();
    let o = b.segments();

    let V0 = a.knots.vector();
    let W0 = b.knots.vector();

    // Adjust left knots
    let Vadj = adjust_knots(p, V0, m, W0);

    // Adjust right knots
    let Vrev = reversed(V0);
    let Wrev = reversed(W0);
    let Wadjrev = adjust_knots(p, &Wrev, o, &Vrev);
    let Wadj = reversed(&Wadjrev).add_scalar(1.);

    (Vadj, Wrev, Wadj)
}

fn adjust_knots(p: usize, Uleft: &VecD, nLeft: usize, Uright: &VecD) -> VecD {
    let mut UleftAdj = VecD::zeros(nLeft + p + 2);

    UleftAdj.head_mut(nLeft + 2).copy_from(&Uleft.head(nLeft + 2));

    UleftAdj.tail_mut(p).copy_from(&Uright.segment(p + 1, p).add_scalar(1.));

    UleftAdj
}

fn create_merged_knot_vector(a: &Curve, b: &Curve, Vadj: &VecD, Wadj: &VecD) -> VecD {
    let m = a.segments();
    let o = b.segments();

    // construct the knot vector of the merged spline
    let mut Umerged = VecD::zeros(m + 2 + o + 1);

    Umerged.head_mut(m + 2).copy_from(&Vadj.head(m + 2));
    Umerged.tail_mut(o + 1).copy_from(&Wadj.tail(o + 1));

    // rescale the knot vector of the merged spline (u in [0,2]) to [0,1] by dividing through the last element
    Umerged.div_assign(Umerged[m + o + 2]);

    Umerged
}

fn generateDerivativeControlPoint(
    idx: usize,
    k: usize,
    Pshifted: &MatD,
    U0: &VecD,
    p: usize,
    n: usize,
    dim: usize,
) -> VecD {
    let mut controlPoint = VecD::zeros(dim);

    assert!(idx <= n - k, "Index out of bounds: only n-k control points exist");
    for zeroOrderIdx in 0..=n {
        controlPoint += prefactor(p, idx, zeroOrderIdx, k, Pshifted, U0) * Pshifted.column(zeroOrderIdx);
    }
    controlPoint
}

fn adjust_shifted_control_points(
    Pshifted: &MatD, //shifted_control_points: &MatD,
    U0: &VecD,       //original_knot_vector: &VecD,
    Uadj: &VecD,     //adjusted_knot_vector: &VecD,
    p: usize,
    n: usize,
    dim: usize,
) -> MatD {
    let mut padj = MatD::zeros(dim, n + 1);
    let mut qki: Vec<Vec<VecD>> = vec![Vec::new(); n + 1];

    for (i, elem) in qki.iter_mut().enumerate().take(n + 1) {
        elem.push(Pshifted.column(i).into());
    }

    for k in 0..=p - 1 {
        let derivative_point = generateDerivativeControlPoint(n - p + 1, k, Pshifted, U0, p, n, dim); //generateDerivativeControlPoint(n - p + 1, k, shifted_control_points, original_knot_vector);
        qki[n - p + 1].push(derivative_point);
    }

    for i in n - p + 2..=n {
        qki[i].resize(n - i + 1, VecD::zeros(dim));
        for k in (0..=n - i).rev() {
            qki[i][k] = ((Uadj[i + p] - Uadj[i + k]) / ((p - k) as f64)) * &qki[i - 1][k + 1] + &qki[i - 1][k];
        }
    }

    for (i, elem) in qki.iter_mut().enumerate().take(n + 1) {
        padj.set_column(i, &elem[0]);
    }

    padj
}

fn adjust_shifted_control_points_of_both_splines(
    a: &Curve,
    Sshifted: &MatD,
    Tshifted: &MatD,
    Vadj: &VecD,
    Wrev: &VecD,
    Wadjrev: &VecD,
) -> (MatD, MatD) {
    let p = a.degree();
    let n = a.segments();
    let dim = a.dimension();

    let V0 = a.knots.vector();

    // obtain adjusted, shifted control points of the second spline
    let Sadj = adjust_shifted_control_points(Sshifted, V0, Vadj, p, n, dim);

    // reverse control points of the second spline
    let P2shiftedrev = curve::points::reversed(Tshifted);

    // obtain adjusted, shifted control points of the reversed second spline
    let Tadjrev = adjust_shifted_control_points(&P2shiftedrev, Wrev, Wadjrev, p, n, dim);

    // obtain adjusted, shifted control points of the second spline
    let Tadj = curve::points::reversed(&Tadjrev);

    (Sadj, Tadj)
}

fn generate_control_point_vector_of_merged_spline(a: &Curve, b: &Curve, Sadj: &MatD, Tadj: &MatD) -> MatD {
    let p = a.degree();
    let dim = a.dimension();
    let n_points1 = a.points.count();
    let n_points2 = b.points.count();

    let mut P_merged = MatD::zeros(dim, n_points1 + n_points2 - p);

    P_merged.columns_mut(0, n_points1).copy_from(Sadj);

    let right_cols = n_points2 + 1 - p;
    P_merged
        .columns_mut(P_merged.ncols() - right_cols, right_cols)
        .copy_from(&Tadj.columns(Tadj.ncols() - right_cols, right_cols));

    P_merged
}

fn kronecker_delta(i: usize, j: usize) -> bool {
    i == j
}

/// `p` degree
/// `i` index
/// `i0` 0th order index
/// `k` derivative order
/// `U0` 0th order knot vector
/// `P0` 0th order control point matrix
fn prefactor(p: usize, i: usize, i0: usize, k: usize, P0: &MatD, U0: &VecD) -> f64 {
    let n = P0.ncols() - 1; // TODO use points

    if i <= n - k {
        if k == 0 {
            if kronecker_delta(i, i0) {
                1.
            } else {
                0.
            }
        } else if U0[i + p + 1] == U0[i + k] {
            0.
        } else {
            (p + 1 - k) as f64 / (U0[i + p + 1] - U0[i + k])
                * (prefactor(p, i + 1, i0, k - 1, P0, U0) - prefactor(p, i, i0, k - 1, P0, U0))
        }
    } else {
        0.
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};

    use crate::curve::{
        generation::{generate, Generation::Manual},
        knots::Generation::Uniform,
        Curve,
    };

    use super::*;

    fn test_bspline(degree: usize, points: MatD) -> Curve {
        generate(Manual { degree, points: ControlPoints::new(points), knots: Uniform }).unwrap()
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

        use crate::curve::points;

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
