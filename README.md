# `bsplines` Rust Library

[![Crates.io](https://img.shields.io/crates/v/bsplines)](https://crates.io/crates/bsplines)
[![Docs.rs](https://docs.rs/bsplines/badge.svg)](https://docs.rs/bsplines)
[![License](https://img.shields.io/crates/l/bsplines)](https://www.apache.org/licenses/LICENSE-2.0)

Rust library for vectorized, N-dimensional B-spline curves and their derivatives based
on [nalgebra](https://docs.rs/nalgebra/latest/nalgebra/).

## Usage

```rust
use bsplines::{Curve, DataPoints};
use nalgebra::dmatrix;

// Five 2D data points, one column per point.
let data = DataPoints::new(dmatrix![
    -2.0,-1.0, 0.0, 1.0, 2.0; // x
     0.5,-0.5, 1.5,-1.5, 0.5; // y
]);

// Interpolate the data with a cubic curve and evaluate it.
let curve = Curve::interpolate(&data, 3)?;
let point = curve.evaluate(0.5)?;
let velocity = curve.evaluate_derivative(0.5, 1)?;

// Or approximate it with a penalized least-squares fit.
let fitted = Curve::fit(&data, 3).loose_ends().penalized(0.5, 2).build()?;
```

Curves can also be built directly from control points (`Curve::with_uniform_knots`,
`Curve::new`) and manipulated afterwards: knot insertion, splitting, merging with
derivative continuity, and reversal. See the [documentation](https://docs.rs/bsplines)
for the full API.

## 🚧 This Library is Under Construction 🚧

- [ ] Use iterators and simplify loops
- [ ] Add benchmarks and improve performance
