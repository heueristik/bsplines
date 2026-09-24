//! Measures how the interpolation and the least-squares fit scale with the number and the dimension of the data
//! points. The report in `target/criterion/report/index.html` plots the times over the number of data points.

use std::{f64::consts::TAU, hint::black_box, time::Duration};

use bsplines::{Curve, DataPoints};
use criterion::{AxisScale, BenchmarkId, Criterion, PlotConfiguration, SamplingMode, criterion_group, criterion_main};
use nalgebra::DMatrix;

const DEGREE: usize = 3;
const DIMENSIONS: [usize; 3] = [1, 10, 100];

/// The fit places the same number of control points for every number of data points.
const FIT_CONTROL_POINTS: usize = 10;

/// Returns data points on a closed curve, with a different phase in each dimension.
fn data_points(dimension: usize, count: usize) -> DataPoints {
    DataPoints::new(DMatrix::from_fn(dimension, count, |row, column| {
        (TAU * column as f64 / (count - 1) as f64 + row as f64).sin()
    }))
}

fn interpolate(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("interpolate");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    // The interpolation solves a dense system with as many unknowns as data points: 1000 points take about a second.
    group.sampling_mode(SamplingMode::Flat).sample_size(10).measurement_time(Duration::from_secs(10));

    for dimension in DIMENSIONS {
        for count in [10, 100, 1_000] {
            let data = data_points(dimension, count);
            group.bench_with_input(
                BenchmarkId::new(format!("dimension {dimension}"), count),
                &data,
                |bencher, data| bencher.iter(|| Curve::interpolate(black_box(data), DEGREE).unwrap()),
            );
        }
    }
    group.finish();
}

fn fit(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("fit");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for dimension in DIMENSIONS {
        for count in [10, 100, 1_000, 10_000, 100_000] {
            let data = data_points(dimension, count);
            group.bench_with_input(
                BenchmarkId::new(format!("dimension {dimension}"), count),
                &data,
                |bencher, data| {
                    bencher.iter(|| {
                        Curve::fit(black_box(data), DEGREE)
                            .polygon_segments(FIT_CONTROL_POINTS - 1)
                            .loose_ends()
                            .penalized(0.5, 2)
                            .build()
                            .unwrap()
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, interpolate, fit);
criterion_main!(benches);
