use crate::{
    curve::{
        parameters::Parameters,
        points::{DataPoints, Points},
    },
    types::VecD,
};

/// Creates a parameters for every data point and distributes them equally ranging from 0 to 1.
/// This method is not recommended, as it can produce erratic shapes (such as loops) when the data is unevenly spaced.
///  see eq. (9.3) in `Piegl1997`
pub fn equally_spaced(segments: usize) -> Parameters {
    let m = segments; //data_points.nrows() - 1;
    let mut u_bar = VecD::zeros(m + 1);

    for g in 1..m {
        u_bar[g] = g as f64 / m as f64;
    }
    u_bar[m] = 1f64;

    Parameters { vector: u_bar, segments }
}

///  see eqs. (9.4) and (9.5) in `Piegl1997`
pub fn centripetal(points: &DataPoints) -> Parameters {
    let m = points.segments(); //data_points.nrows() - 1;

    let mut sum = 0.0;

    for g in 1..=m {
        let diff = points.get(g) - points.get(g - 1);
        sum += diff.norm().sqrt()
    }

    debug_assert!(sum >= f64::EPSILON.sqrt(), "sum {} is too small. Use the equally spaced methods instead", sum);

    let mut u_bar = VecD::zeros(m + 1);

    for g in 1..m {
        let diff = points.get(g) - points.get(g - 1);
        u_bar[g] = u_bar[g - 1] + diff.norm().sqrt() / sum;
    }

    u_bar[m] = 1.0;

    Parameters { vector: u_bar, segments: points.segments() }
}

pub fn chord_length(points: &DataPoints) -> Parameters {
    let m = points.segments();

    let mut sum = 0f64;

    for g in 1..=m {
        let diff = points.get(g) - points.get(g - 1);
        sum += diff.norm();
    }

    let mut u_bar = VecD::zeros(m + 1);
    for g in 1..m {
        let diff = points.get(g) - points.get(g - 1);
        u_bar[g] = u_bar[g - 1] + diff.norm() / sum;
    }

    u_bar[m] = 1f64;

    Parameters { vector: u_bar, segments: points.segments() }
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};

    use super::*;

    mod equally_spaced {
        use super::*;

        #[test]
        fn test() {
            let points = DataPoints::new(dmatrix![1.0, 2.0, 3.0, 4.0, 5.0;]);
            let knots = equally_spaced(points.segments());
            assert_eq!(knots.vector, dvector![0., 0.25, 0.5, 0.75, 1.]);
        }
    }

    mod chord_length {
        use super::*;

        #[test]
        fn linear_1() {
            let points = DataPoints::new(dmatrix![1.0, 2.0, 3.0, 4.0, 5.0;]);
            let knots = chord_length(&points);
            assert_eq!(knots.vector, dvector![0., 0.25, 0.5, 0.75, 1.]);
        }

        #[test]
        fn linear_2() {
            let points = DataPoints::new(dmatrix![1.0, 3.0, 5.0;]);
            let knots = chord_length(&points);
            assert_eq!(knots.vector, dvector![0., 0.5, 1.]);
        }

        #[test]
        fn non_linear_1() {
            let points = DataPoints::new(dmatrix![1.0, 2.0, 5.0;]);
            let knots = chord_length(&points);
            assert_eq!(knots.vector, dvector![0., 0.25, 1.]);
        }

        #[test]
        fn non_linear_2() {
            let points = DataPoints::new(dmatrix![1.0, 4.0, 5.0;]);
            let knots = chord_length(&points);
            assert_eq!(knots.vector, dvector![0., 0.75, 1.]);
        }
    }

    mod centripetal {
        use super::*;

        #[test]
        fn linear_1() {
            let points = DataPoints::new(dmatrix![1.0, 2.0, 3.0, 4.0, 5.0;]);
            let knots = centripetal(&points);
            assert_eq!(knots.vector, dvector![0., 0.25, 0.5, 0.75, 1.]);
        }

        #[test]
        fn linear_2() {
            let points = DataPoints::new(dmatrix![1.0, 3.0, 5.0;]);
            let knots = centripetal(&points);
            assert_eq!(knots.vector, dvector![0., 0.5, 1.]);
        }

        #[test]
        fn non_linear_1() {
            let points = DataPoints::new(dmatrix![1.0, 2.0, 11.0;]);
            let knots = centripetal(&points);
            assert_eq!(knots.vector, dvector![0., 0.25, 1.]);
        }

        #[test]
        fn non_linear_2() {
            let points = DataPoints::new(dmatrix![1.0, 10.0, 11.0;]);
            let knots = centripetal(&points);
            assert_eq!(knots.vector, dvector![0., 0.75, 1.]);
        }
    }
}
