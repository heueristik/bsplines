use nalgebra::DMatrix;

use crate::{
    error::{Error, Result},
    knots::Knots,
    parameters::Parameters,
    points::{DataPoints, Points},
    svd::decompose,
};

pub fn interpolate(knots: &Knots, points: &DataPoints, parameters: &Parameters) -> Result<DMatrix<f64>> {
    let polyline_segments = points.polyline_segments();

    let u_bar = parameters.vector();

    // Interpolation uses one control point per data point, so the system is square.
    let mut basis_matrix = DMatrix::zeros(points.count(), points.count());
    for i in 0..=polyline_segments {
        for g in 0..=polyline_segments {
            basis_matrix[(g, i)] = knots.basis_of_derivative_curve(0, i, u_bar[g]);
        }
    }

    // A basis function that is zero at its own parameter makes the system singular (Schoenberg-Whitney).
    if let Some(index) = (0..=polyline_segments).find(|&i| basis_matrix[(i, i)] == 0.0) {
        return Err(Error::SingularInterpolation { index });
    }

    let svd = decompose(basis_matrix.clone())?;
    // A threshold relative to the largest singular value keeps the small but valid singular values.
    let threshold = svd.singular_values.max() * points.count() as f64 * f64::EPSILON;
    let transposed_data = points.matrix().transpose();
    let solution = svd.solve(&transposed_data, threshold).expect("the SVD was computed with both U and V^T");

    // A nearly singular system loses the data points that its smallest singular values carry.
    let residuals = &basis_matrix * &solution - &transposed_data;
    let tolerance = f64::EPSILON.sqrt() * (1.0 + transposed_data.amax());
    if let Some(index) = residuals.row_iter().position(|residual| residual.amax() > tolerance) {
        return Err(Error::SingularInterpolation { index });
    }
    Ok(solution.transpose())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector};

    use crate::{
        Curve,
        knots::KnotMethod::{Averaging, Uniform},
        parameters::ParameterMethod::ChordLength,
        points::ControlPoints,
    };

    use super::*;

    /// Returns parameters whose fourth and fifth values lie the given distance behind the uniform knots
    /// 0.25 and 0.5, where the basis functions 3 and 4 of degree 2 start.
    fn parameters_near_the_knots(distance: f64) -> Parameters {
        Parameters::new(dvector![0.0, 0.1, 0.2, 0.25 + distance, 0.5 + distance, 1.0]).unwrap()
    }

    fn zigzag() -> DataPoints {
        DataPoints::new(dmatrix![
            0., 1.,  2., 3.,  4., 5.;
            0., 1., -1., 1., -1., 0.;
        ])
    }

    #[test]
    fn interpolate_errors_for_a_nearly_singular_system() {
        let parameters = parameters_near_the_knots(1e-6);
        let knots = Knots::generate(2, 5, &parameters, Uniform).unwrap();
        assert!(knots.basis(3, parameters.vector()[3]).unwrap() > 0.0, "no basis function is zero at its parameter");

        assert!(matches!(interpolate(&knots, &zigzag(), &parameters), Err(Error::SingularInterpolation { .. })));
    }

    #[test]
    fn interpolate_passes_through_the_data_of_an_ill_conditioned_system() {
        let parameters = parameters_near_the_knots(1e-3);
        let knots = Knots::generate(2, 5, &parameters, Uniform).unwrap();
        let data = zigzag();

        let control_points = interpolate(&knots, &data, &parameters).unwrap();

        let curve = Curve::new(knots, ControlPoints::new(control_points)).unwrap();
        for (g, &u) in parameters.vector().iter().enumerate() {
            assert_relative_eq!(curve.evaluate(u).unwrap(), data.matrix().column(g).clone_owned(), epsilon = 1e-6);
        }
    }

    #[test]
    fn linear() {
        let points = DataPoints::new(dmatrix![
            1., 2., 3., 4.;
            1., 2., 3., 4.;
        ]);

        let parameters = Parameters::generate(&points, ChordLength).unwrap();
        let knots = Knots::generate(1, points.polyline_segments(), &parameters, Averaging).unwrap();

        assert_eq!(interpolate(&knots, &points, &parameters).unwrap(), *points.matrix());
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
            interpolate(&knots, &points, &parameters).unwrap(),
            dmatrix![
                1., 1.75, 3.25, 4.;
                1., 1.75, 3.25, 4.;
            ],
            epsilon = f64::EPSILON.sqrt()
        );
    }
}
