//! Splits a curve into two independent curves.

use nalgebra::{DMatrix, DVector};

use crate::{
    Curve,
    error::{Error, Result},
    knots::{Knots, normalize},
    manipulation::insert::insert,
    points::{ControlPoints, Points},
    vector_views::VectorViews,
};

/// Splits the curve into two independent curves at the parameter `u` and normalizes both knot vectors.
pub(crate) fn split(curve: &Curve, u: f64) -> Result<(Curve, Curve)> {
    if u <= 0.0 || u >= 1.0 {
        return Err(Error::OutsideDomainInterior { u, min: 0.0, max: 1.0 });
    }

    let degree = curve.degree();

    let mut inserted = curve.clone();

    let span = curve.knots.find_span(u, 0);
    let multiplicity = curve.knots.vector().iter().skip(span).take_while(|&&x| x == u).count();

    if multiplicity > degree {
        return Err(Error::MultiplicityExceedsDegree { u, multiplicity, degree });
    }

    for _ in 0..degree - multiplicity {
        insert(&mut inserted, u)?;
    }

    let knots = inserted.knots.vector();
    let points = inserted.control_points.matrix();

    if multiplicity > 0 {
        let left = {
            let mut left_knots = DVector::zeros(span + degree + 1);
            left_knots.head_mut(span + degree).copy_from(&knots.head(span + degree));
            left_knots[span + degree] = u;

            normalize(&mut left_knots);
            let point_count = left_knots.len() - (degree + 2) + 1;
            let left_points: DMatrix<f64> = points.columns(0, point_count).into();

            Curve::new(Knots::new(degree, left_knots), ControlPoints::new(left_points))?
        };

        let right = {
            let mut right_knots = DVector::zeros(knots.len() + 1 - span);
            right_knots[0] = u;
            right_knots.tail_mut(knots.len() - span).copy_from(&knots.tail(knots.len() - span));

            normalize(&mut right_knots);
            let point_count = right_knots.len() - (degree + 2) + 1;
            let right_points: DMatrix<f64> = points.columns(points.ncols() - point_count, point_count).into();

            Curve::new(Knots::new(degree, right_knots), ControlPoints::new(right_points))?
        };
        Ok((left, right))
    } else {
        let left = {
            let mut left_knots = DVector::zeros(span + degree + 1 + 1);
            left_knots.head_mut(span + degree + 1).copy_from(&knots.head(span + degree + 1));
            left_knots[span + degree + 1] = u;

            normalize(&mut left_knots);

            let point_count = left_knots.len() + 1 - (degree + 2);
            let left_points: DMatrix<f64> = points.columns(0, point_count).into();

            Curve::new(Knots::new(degree, left_knots), ControlPoints::new(left_points))?
        };

        let right = {
            let mut right_knots = DVector::zeros(knots.len() + 1 - (span + 1));
            right_knots[0] = u;
            right_knots.tail_mut(knots.len() - (span + 1)).copy_from(&knots.tail(knots.len() - (span + 1)));

            normalize(&mut right_knots);
            let point_count = right_knots.len() + 1 - (degree + 2);
            let right_points: DMatrix<f64> = points.columns(points.ncols() - point_count, point_count).into();

            Curve::new(Knots::new(degree, right_knots), ControlPoints::new(right_points))?
        };
        Ok((left, right))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector};
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    /// A one-dimensional, linear test curve with default degree two.
    fn curve(#[default(2)] degree: usize) -> Curve {
        let curve = Curve::with_uniform_knots(ControlPoints::new(dmatrix![1., 2., 3., 4., 5., 6.;]), degree).unwrap();
        assert_eq!(curve.knots.vector(), &dvector![0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1.]);
        curve
    }

    #[rstest]
    fn cannot_split_start(curve: Curve) {
        let u = 0.0;
        let result = split(&curve, u);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Error::OutsideDomainInterior { u, min: 0.0, max: 1.0 });
    }

    #[rstest]
    fn cannot_split_end(curve: Curve) {
        let u = 1.0;
        let result = split(&curve, u);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Error::OutsideDomainInterior { u, min: 0.0, max: 1.0 });
    }

    #[rstest]
    fn split_close_to_start(curve: Curve) {
        let eps = f64::EPSILON;
        let (left, right) = split(&curve, 0.0 + eps).unwrap();

        assert_relative_eq!(left.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.], epsilon = eps.sqrt());
        assert_relative_eq!(left.control_points.matrix(), &dmatrix![1., 1., 1.;], epsilon = eps.sqrt());

        assert_relative_eq!(right.knots.vector(), curve.knots.vector(), epsilon = eps.sqrt());
        assert_relative_eq!(right.control_points.matrix(), curve.control_points.matrix(), epsilon = eps.sqrt());
    }

    #[rstest]
    fn split_close_to_end(curve: Curve) {
        let eps = f64::EPSILON;
        let (left, right) = split(&curve, 1.0 - eps).unwrap();

        assert_relative_eq!(left.knots.vector(), curve.knots.vector(), epsilon = eps.sqrt());
        assert_relative_eq!(left.control_points.matrix(), curve.control_points.matrix(), epsilon = eps.sqrt());

        assert_relative_eq!(right.knots.vector(), &dvector![0., 0., 0., 1., 1., 1.], epsilon = eps.sqrt());
        assert_relative_eq!(right.control_points.matrix(), &dmatrix![6., 6., 6.;], epsilon = eps.sqrt());
    }

    #[rstest]
    fn normalized(curve: Curve) {
        let (left, right) = split(&curve, 0.5).unwrap();
        assert_eq!(left.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);
        assert_eq!(left.control_points.matrix(), &dmatrix![1., 2., 3., 3.5;]);

        assert_eq!(right.knots.vector(), &dvector![0., 0., 0., 0.5, 1., 1., 1.]);
        assert_eq!(right.control_points.matrix(), &dmatrix![3.5, 4., 5., 6.;]);
    }
}
