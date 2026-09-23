# B-Splines

A library for N-dimensional B-spline curves and their derivatives, implementing the methods of Piegl & Tiller's *The NURBS Book* (Piegl1997).

## Language

**Curve**:
A parametric B-spline function mapping the domain into N-dimensional space, defined by a knot vector and control points.
_Avoid_: spline (ambiguous: also names the basis functions and surfaces)

**Knot vector (U)**:
The non-decreasing sequence of parameter values that partitions the domain and defines the basis functions.
_Avoid_: knots as a synonym for parameters

**Control points (P)**:
The points whose combination weighted by the basis functions traces the curve. Together they form the control polygon.
_Avoid_: conflating with data points

**Data points**:
Input points that a curve is interpolated through or fitted to. Data points are consumed by curve generation; they are not part of the resulting curve.
_Avoid_: sample points, measurements

**Parameters (ū)**:
The parameter values assigned to the data points for interpolation and fitting, one per data point.
_Avoid_: knots, sites

**Parameter (u)**:
The scalar in the domain at which a curve is evaluated.
_Avoid_: t, time

**Polygon segments (n)**:
The number of segments of the control polygon — one less than the number of control points. Not the number of polynomial pieces of the curve (that is n − p + 1).
_Avoid_: segments (bare), polynomial segments

**Polyline segments (m)**:
The number of chords of the data polyline — one less than the number of data points.
_Avoid_: segments (bare), data segments

**Degree (p)**:
The polynomial degree of the basis functions.
_Avoid_: order (in parts of the literature, order means degree + 1)

**Derivative order (k)**:
Selects which derivative of the curve is meant; the zeroth derivative (k = 0) is the curve itself.
_Avoid_: degree (reserved for the polynomial degree)

**Penalization strength (λ)**:
The weight of the penalty term in a penalized least-squares fit; zero turns the penalty off.
_Avoid_: lambda, smoothing parameter (parameter means u or ū here)

**Difference order (κ)**:
The order of the finite differences between neighboring control points that the penalty term sums, see Eilers1996.
_Avoid_: kappa

**Domain**:
The knot interval on which the curve is defined, spanning from knot p to knot n + 1.

**Clamped**:
A knot vector whose first and last knot values are each repeated p + 1 times, so the curve starts and ends at its end control points.

**Normalized**:
A knot vector whose domain is exactly [0, 1].
_Avoid_: normed
