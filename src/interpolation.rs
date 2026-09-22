use nalgebra::{DMatrix, SVD};

use crate::{
    knots::Knots,
    parameters::Parameters,
    points::{DataPoints, Points},
};

pub fn interpolate(knots: &Knots, points: &DataPoints, parameters: &Parameters) -> DMatrix<f64> {
    let polyline_segments = points.polyline_segments();

    let u_bar = parameters.vector();

    // Interpolation uses one control point per data point, so the system is square.
    let mut basis_matrix = DMatrix::zeros(points.count(), points.count());
    for i in 0..=polyline_segments {
        for g in 0..=polyline_segments {
            basis_matrix[(g, i)] = knots.basis_of_derivative_curve(0, i, u_bar[g]);
        }
    }

    let svd = SVD::new(basis_matrix, true, true);
    let transposed_data = points.matrix().transpose();
    svd.solve(&transposed_data, f64::EPSILON.sqrt()).expect("the SVD was computed with both U and V^T").transpose()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::dmatrix;

    use crate::{knots::KnotMethod::Averaging, parameters::ParameterMethod::ChordLength};

    use super::*;

    #[test]
    fn linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4.;
            1., 2., 3., 4.;
        ]);

        let parameters = Parameters::generate(&points, ChordLength).unwrap();
        let knots = Knots::generate(1, points.polyline_segments(), &parameters, Averaging).unwrap();

        assert_eq!(interpolate(&knots, &points, &parameters), *points.matrix());
    }

    #[test]
    fn quadratic() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4.;
            1., 2., 3., 4.;
        ]);
        let parameters = Parameters::generate(&points, ChordLength).unwrap();
        let knots = Knots::generate(2, points.polyline_segments(), &parameters, Averaging).unwrap();

        assert_relative_eq!(
            interpolate(&knots, &points, &parameters),
            dmatrix![
                1., 1.75, 3.25, 4.;
                1., 1.75, 3.25, 4.;
            ],
            epsilon = f64::EPSILON.sqrt()
        );
    }
}
