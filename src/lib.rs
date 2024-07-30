#![cfg_attr(not(feature = "std"), no_std)]

//! microcheby is a crate for computing and evaluating polynomial approximations
//! of functions of one variable using using [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials).
//! The code is `no_std` compatible, does not depend on `alloc` and is optimized for resource
//! constrained environments where every clock cycle counts. Optimizations include:
//!
//! * [Clenshaw recursion](https://en.wikipedia.org/wiki/Clenshaw_algorithm) for evaluating approximations.
//! * Efficient loop free functions for evaluating low order approximations.
//! * Even more efficient loop free evaluation if the range happens to be [-1, 1].
//! * Approximation evaluation without divisions.
//!
//! # Installing
//!
//! Add the following line to your Cargo.toml file:
//!
//! ```text
//! microcheby = "0.1"
//! ```
//!
//! To use microcheby in a `no_std` environment:
//!
//! ```text
//! microcheby = { version = "0.1", default-features = false }
//! ```
//!
//! # Chebyshev approximation
//!
//! Sufficiently well behaved functions can be expressed as an infinite weighted
//! sum of so called [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) of increasing order.
//! Such a sum is known as a Chebyshev expansion.
//! If the target function is smooth enough, the coefficients (weights) of the expansion will
//! typically converge to zero quickly and only the first few terms are needed to get a good approximation.
//! For a truncated expansion with _n_ terms, an estimate of the approximation error is given by
//! the magnitude of coefficient _n+1_.
//!
//! For a more detailed introduction, see [5.8 Chebyshev Approximation](http://www.ff.bg.ac.rs/Katedre/Nuklearna/SiteNuklearna/bookcpdf/c5-8.pdf) in Numerical Recipes in C: The Art of Scientific Computing.
//!
//! # Basic usage
//!
//! ```
//! use microcheby::ChebyshevExpansion;
//!
//! // Compute a 6 term expansion approximating the square root of x on the interval [0.1, 1.0]
//! let sqrt_exp = ChebyshevExpansion::<6>::fit(0.1, 1.0, |x| x.sqrt());
//! // Get the approximated value at x=0.7
//! let value_approx = sqrt_exp.eval(0.7);
//! // Get the actual value at x=0.7
//! let value_actual = 0.7_f32.sqrt();
//! // Compute the approximation error
//! let error = value_actual - value_approx;
//! assert!(error.abs() < 0.0002);
//! ```
//!
//! # Precomputing and instantiating expansions
//!
//! Computing expansion coefficients requires an accurate cosine function and potentially costly
//! target function evaluations, so it is sometimes desirable to precompute the coefficients
//! and then use them to instantiate the expansion.
//!
//! ```
//! use microcheby::ChebyshevExpansion;
//!
//! // Compute a 6 term expansion approximating the square root of x on the interval [0.1, 1.0]
//! let sqrt_exp = ChebyshevExpansion::<6>::fit(0.1, 1.0, |x| x.sqrt());
//!
//! // x_min, x_max and coeffs are needed to instantiate the expansion.
//! // You can either print them to the terminal like this or use the associated getters.
//! println!("{sqrt_exp}");
//!
//! // Instantiate the expansion computed above using arguments copied from the terminal.
//! let exp = ChebyshevExpansion::new(
//!     0.1,
//!     1.0,
//!     [1.4066492, 0.32991815, -0.04125017, 0.010474294, -0.0032901317, 0.0010244437]
//! );
//! assert_eq!(sqrt_exp.x_min(), exp.x_min());
//! assert_eq!(sqrt_exp.x_max(), exp.x_max());
//! assert_eq!(sqrt_exp.coeffs(), exp.coeffs());
//! ```
//!
//! Expansions can also be instantiated statically at compile time.
//!
//! ```
//! use microcheby::ChebyshevExpansion;
//!
//! // Compute a 6 term expansion approximating the square root of x on the interval [0.1, 1.0]
//! let sqrt_exp = ChebyshevExpansion::<6>::fit(0.1, 1.0, |x| x.sqrt());
//!
//! // x_min, range_scale and coeffs_internal are needed to instantiate the expansion
//! // at compile time since floating point operations are not allowed in constant expressions.
//! // You can either print them to the terminal like this or get them using a debugger.
//! println!("{:?}", sqrt_exp);
//!
//! // A statically allocated ChebyshevExpansion
//! const EXP:ChebyshevExpansion<6> = ChebyshevExpansion::const_new(
//!     0.1,
//!     4.4444447,
//!     [0.7033246, 0.32991815, -0.04125017, 0.010474294, -0.0032901317, 0.0010244437]
//! );
//! assert_eq!(sqrt_exp.x_min(), EXP.x_min());
//! assert_eq!(sqrt_exp.x_max(), EXP.x_max());
//! assert_eq!(sqrt_exp.coeffs(), EXP.coeffs());
//! ```

use core::f32::consts::PI;

#[cfg(feature = "std")]
use core::fmt;

/// Boundary matching is the process of adding a linear
/// term _ax + b_ to the Chebyshev expansion to make the it
/// match the original function exactly at `x_min` and/or `x_max`.
/// This probably increases the approximation error, but
/// can be useful when exact matches at `x_min` and/or `x_max` are
/// more important than overall accuracy.
#[derive(PartialEq)]
pub enum MatchBoundary {
    /// No boundary matching
    None,
    /// Make the expansion match the original function at `x_min`.
    Min,
    /// Make the expansion match the original function at `x_max`.
    Max,
    /// Make the expansion match the original function at `x_min` and `x_max`.
    Both,
}

/// An N term Chebyshev expansion.
/// See [5.8 Chebyshev Approximation](http://www.ff.bg.ac.rs/Katedre/Nuklearna/SiteNuklearna/bookcpdf/c5-8.pdf) in Numerical Recipes in C: The Art of Scientific Computing.
#[derive(Clone, Copy)]
pub struct ChebyshevExpansion<const N: usize> {
    /// Chebyshev expansion coefficients, except that the first coefficient has been multiplied by 0.5.
    coeffs_internal: [f32; N],
    /// The start of the range to approximate.
    x_min: f32,
    /// A precomputed constant used to speed up evaluation. Equals `4.0 / (x_max - x_min)`.
    range_scale: f32,
}

#[cfg(feature = "std")]
impl<const N: usize> fmt::Display for ChebyshevExpansion<N> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        writeln!(fmt, "ChebyshevExpansion {{")?;
        writeln!(fmt, "  x_min: {},", self.x_min())?;
        writeln!(fmt, "  x_max: {},", self.x_max())?;
        writeln!(fmt, "  coeffs: {:?}", self.coeffs())?;
        writeln!(fmt, "}}")?;
        Ok(())
    }
}

#[cfg(feature = "std")]
impl<const N: usize> fmt::Debug for ChebyshevExpansion<N> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        writeln!(fmt, "ChebyshevExpansion {{")?;
        writeln!(fmt, "  x_min: {},", self.x_min())?;
        writeln!(fmt, "  range_scale: {},", self.range_scale)?;
        writeln!(fmt, "  coeffs_internal: {:?}", self.coeffs_internal)?;
        writeln!(fmt, "}}")?;
        Ok(())
    }
}

impl<const N: usize> ChebyshevExpansion<N> {
    /// Creates an `N` term `ChebyshevExpansion` instance approximating a given function on a given range.
    ///
    /// # Arguments
    /// * `x_min` The start of the range to approximate.
    /// * `x_max` The end of the range to approximate.
    /// * `f` The function to approximate.
    #[cfg(feature = "std")]
    pub fn fit<F>(x_min: f32, x_max: f32, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        ChebyshevExpansion::fit_with_options(x_min, x_max, f, |x: f32| x.cos(), MatchBoundary::None)
    }

    /// Creates an `N` term `ChebyshevExpansion` instance approximating a given function on a given range,
    /// optionally performing boundary matching.
    ///
    /// # Arguments
    /// * `x_min` - The start of the range to approximate.
    /// * `x_max` - The end of the range to approximate.
    /// * `f` - The function to approximate.
    /// * `cos` - A function for computing _cos(x)_. Allows the caller to provide a
    /// custom cosine implementation if the standard one is not available, which is the case in `no_std` environments.
    /// * `match_boundary` - Indicates if the expansion should be altered so that the approximation matches the given function at `x_min` and/or `x_max`.
    pub fn fit_with_options<F, G>(
        x_min: f32,
        x_max: f32,
        f: F,
        cos: G,
        match_boundary: MatchBoundary,
    ) -> Self
    where
        F: Fn(f32) -> f32,
        G: Fn(f32) -> f32,
    {
        let mut approx = ChebyshevExpansion::new(x_min, x_max, [0.0; N]);
        let n_inv = 1.0 / (N as f32);
        for (j, c_j) in approx.coeffs_internal.iter_mut().enumerate() {
            for k in 0..N {
                let x_rel = 0.5 * (1.0 + cos(PI * (k as f32 + 0.5) * n_inv));
                let x = x_min + (x_max - x_min) * x_rel;
                let f_val = f(x);
                let weight = cos(PI * (j as f32) * (k as f32 + 0.5) * n_inv);
                *c_j += 2.0 * f_val * weight * n_inv
            }
        }

        // "pre-baking" multiply by 0.5 to avoid it during evaluation
        approx.coeffs_internal[0] *= 0.5;

        if match_boundary != MatchBoundary::None && N > 1 {
            // Add a linear term a + bx that offsets the left and right
            // ends to the desired values
            let (x_min_offs, x_max_offs) = match match_boundary {
                MatchBoundary::Min => (f(x_min) - approx.eval(x_min), 0.0),
                MatchBoundary::Max => (0.0, f(x_max) - approx.eval(x_max)),
                MatchBoundary::Both => {
                    (f(x_min) - approx.eval(x_min), f(x_max) - approx.eval(x_max))
                }
                _ => (0.0, 0.0),
            };
            let a = 0.5 * (x_max_offs + x_min_offs);
            let b = 0.5 * (x_max_offs - x_min_offs);
            approx.coeffs_internal[0] += a; // multiplied by 2.0 * 0.5 = 1 due to c0 pre-bake multiply above
            approx.coeffs_internal[1] += b;
        }

        approx
    }

    /// Create an `N` term `ChebyshevExpansion` instance using a given range and coefficients.
    /// The values for the arguments can be accessed by the associated getter function.
    ///
    /// # Arguments
    ///
    /// * `x_min` - The start of the range to approximate.
    /// * `x_max` - The end of the range to approximate.
    /// * `coeffs` - Chebyshev expansion coefficients
    pub fn new(x_min: f32, x_max: f32, coeffs: [f32; N]) -> Self {
        let mut coeffs_internal = coeffs;
        coeffs_internal[0] *= 0.5;
        ChebyshevExpansion::const_new(x_min, 4.0 / (x_max - x_min), coeffs_internal)
    }

    /// Creates an `N` term `ChebyshevExpansion` instance statically. Due to floating point
    /// limitations in const expressions, `range_scale` and `coeffs_internal` must be passed "raw".
    /// The values for these can either be accessed using a debugger or printed using
    /// ```text
    /// println!("{exp:?}")
    /// ```
    /// and then copy-pasted (`exp` is a `ChebyshevExpansion` instance).
    ///
    /// # Arguments
    ///
    /// * `x_min` - The start of the approximated function's range.
    /// * `range_scale` - Equal to `4.0 / (x_max - x_min)`, where `x_max` is the end of the approximated function's range.
    /// * `coeffs_internal` - The same as the Chebychev weights, except that the element at index 0 has been divided by 2.
    pub const fn const_new(x_min: f32, range_scale: f32, coeffs_internal: [f32; N]) -> Self {
        ChebyshevExpansion {
            x_min,
            coeffs_internal,
            range_scale,
        }
    }

    /// The start of the range to approximate.
    pub fn x_min(&self) -> f32 {
        self.x_min
    }

    /// The end of the range to approximate.
    pub fn x_max(&self) -> f32 {
        4.0 / self.range_scale + self.x_min
    }

    /// The coefficients of the Chebyshev expansion.
    pub fn coeffs(&self) -> [f32; N] {
        let mut coeffs = self.coeffs_internal;
        if N > 0 {
            // compensate for 'pre baked' multiplication by 0.5
            coeffs[0] *= 2.0;
        }
        coeffs
    }

    /// Evaluates the Chebyshev expansion at a given point.
    ///
    /// # Arguments
    ///
    /// * `x` - Evaluate the expansion at this x value.
    pub fn eval(&self, x: f32) -> f32 {
        let x_rel_2 = -2.0 + (x - self.x_min) * self.range_scale;
        let mut d = 0.0;
        let mut dd = 0.0;
        let mut temp;

        for cj in self.coeffs_internal.iter().skip(1).rev() {
            temp = d;
            d = x_rel_2 * d - dd + cj;
            dd = temp;
        }

        0.5 * x_rel_2 * d - dd + self.coeffs_internal[0]
    }

    /// Evaluates the first `n` terms of the Chebyshev expansion at a given point.
    ///
    /// # Arguments
    ///
    /// * `x` - Evaluate the expansion at this x value.
    /// * `n` - Evaluate this many terms.
    pub fn eval_trunc(&self, x: f32, n: usize) -> f32 {
        let x_rel_2 = -2.0 + (x - self.x_min) * self.range_scale;
        let mut d = 0.0;
        let mut dd = 0.0;
        let mut temp;

        for cj in self.coeffs_internal.iter().take(n).skip(1).rev() {
            temp = d;
            d = x_rel_2 * d - dd + cj;
            dd = temp;
        }

        0.5 * x_rel_2 * d - dd + self.coeffs_internal[0]
    }

    /// Checks if the Chebyshev expansion is odd with a given tolerance,
    /// i.e if coefficients 0, 2, 4... are (close to) zero.
    ///
    /// # Arguments
    /// * `eps` - Threat coefficients with an absolute value less than this as being zero.
    pub fn is_odd(&self, eps: f32) -> bool {
        for ci in self.coeffs_internal.iter().step_by(2) {
            if ci.abs() > eps {
                return false;
            }
        }

        true
    }

    /// Checks if the Chebyshev expansion is even with a given tolerance,
    /// i.e if coefficients 1, 3, 5... are (close to) zero.
    ///
    /// # Arguments
    /// * `eps` - Threat coefficients with an absolute value less than this as being zero.
    pub fn is_even(&self, eps: f32) -> bool {
        for (i, ci) in self.coeffs_internal.iter().skip(1).step_by(2).enumerate() {
            let scale = if i == 0 { 2.0 } else { 1.0 }; // account for "baked" c0 scale of 0.5
            if scale * ci.abs() > eps {
                return false;
            }
        }

        true
    }

    /// Returns a `ChebyshevExpansion` instance containing the `M` first coefficients.
    pub fn trunc<const M: usize>(&self) -> ChebyshevExpansion<M> {
        assert!(M <= N);
        let mut coeffs_internal = [0.0; M];
        for (ci_trunc, ci) in coeffs_internal.iter_mut().zip(self.coeffs_internal.iter()) {
            *ci_trunc = *ci;
        }
        ChebyshevExpansion::const_new(self.x_min, self.range_scale, coeffs_internal)
    }
}

impl ChebyshevExpansion<2> {
    /// Optimized, loop free evaluation of two term expansions.
    ///
    /// # Arguments
    /// * `x` - Evaluate the expansion at this x value.
    pub fn eval_2(&self, x: f32) -> f32 {
        let x_rel_2 = -2.0 + (x - self.x_min) * self.range_scale;
        0.5 * x_rel_2 * self.coeffs_internal[1] + self.coeffs_internal[0]
    }

    /// Optimized, loop free evaluation of two term expansions
    /// where `x_min` is -1 and `x_max` is 1, which enables further optimizations.
    ///
    /// # Arguments
    /// * `x` - Evaluate the expansion at this x value. It is assumed that `-1.0 <= x <= 1.0`.
    pub fn eval_2_neg1_to_1(&self, x: f32) -> f32 {
        let x_rel_2 = 2.0 * x;
        0.5 * x_rel_2 * self.coeffs_internal[1] + self.coeffs_internal[0]
    }
}

impl ChebyshevExpansion<3> {
    /// Optimized, loop free evaluation of three term expansions.
    ///
    /// # Arguments
    /// * `x` - Evaluate the expansion at this x value.
    pub fn eval_3(&self, x: f32) -> f32 {
        let x_rel_2 = -2.0 + (x - self.x_min) * self.range_scale;
        let d_2 = self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }

    /// Optimized, loop free evaluation of three term expansions
    /// where `x_min` is -1 and `x_max` is 1, which enables further optimizations.
    ///
    /// # Arguments
    /// * `x` - Evaluate the expansion at this x value. It is assumed that `-1.0 <= x <= 1.0`.
    pub fn eval_3_neg1_to_1(self, x: f32) -> f32 {
        let x_rel_2 = 2.0 * x;
        let d_2 = self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }
}

impl ChebyshevExpansion<4> {
    /// Optimized, loop free evaluation of four term expansions.
    ///
    /// # Arguments
    /// * `x` - Evaluate the expansion at this x value.
    pub fn eval_4(&self, x: f32) -> f32 {
        let x_rel_2 = -2.0 + (x - self.x_min) * self.range_scale;
        let d_3 = self.coeffs_internal[3];
        let d_2 = x_rel_2 * d_3 + self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 - d_3 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }

    /// Optimized, loop free evaluation of four term expansions
    /// where `x_min` is -1 and `x_max` is 1, which enables further optimizations.
    ///
    /// # Arguments
    /// * `x` - Evaluate the expansion at this x value. It is assumed that `-1.0 <= x <= 1.0`.
    pub fn eval_4_neg1_to_1(&self, x: f32) -> f32 {
        let x_rel_2 = 2.0 * x;
        let d_3 = self.coeffs_internal[3];
        let d_2 = x_rel_2 * d_3 + self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 - d_3 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }
}

impl ChebyshevExpansion<5> {
    /// Optimized, loop free evaluation of five term expansions.
    ///
    /// # Arguments
    /// * `x` - Evaluate the expansion at this x value.
    pub fn eval_5(&self, x: f32) -> f32 {
        let x_rel_2 = -2.0 + (x - self.x_min) * self.range_scale;
        let d_4 = self.coeffs_internal[4];
        let d_3 = x_rel_2 * d_4 /*-0*/+ self.coeffs_internal[3];
        let d_2 = x_rel_2 * d_3 - d_4 + self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 - d_3 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }

    /// Optimized, loop free evaluation of five term expansions
    /// where `x_min` is -1 and `x_max` is 1, which enables further optimizations.
    ///
    /// # Arguments
    /// * `x` - Evaluate the expansion at this x value. It is assumed that `-1.0 <= x <= 1.0`.
    pub fn eval_5_neg1_to_1(&self, x: f32) -> f32 {
        let x_rel_2 = 2.0 * x;
        let d_4 = self.coeffs_internal[4];
        let d_3 = x_rel_2 * d_4 /*-0*/+ self.coeffs_internal[3];
        let d_2 = x_rel_2 * d_3 - d_4 + self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 - d_3 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }
}

impl ChebyshevExpansion<6> {
    /// Optimized, loop free evaluation of six term expansions.
    ///
    /// # Arguments
    /// * `x` - Evaluate the expansion at this x value.
    pub fn eval_6(&self, x: f32) -> f32 {
        let x_rel_2 = -2.0 + (x - self.x_min) * self.range_scale;
        let d_5 = self.coeffs_internal[5];
        let d_4 = x_rel_2 * d_5 + self.coeffs_internal[4];
        let d_3 = x_rel_2 * d_4 - d_5 + self.coeffs_internal[3];
        let d_2 = x_rel_2 * d_3 - d_4 + self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 - d_3 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }

    /// Optimized, loop free evaluation of six term expansions
    /// where `x_min` is -1 and `x_max` is 1, which enables further optimizations.
    ///
    /// # Arguments
    /// * `x` - Evaluate the expansion at this x value. It is assumed that `-1.0 <= x <= 1.0`.
    pub fn eval_6_neg1_to_1(&self, x: f32) -> f32 {
        let x_rel_2 = 2.0 * x;
        let d_5 = self.coeffs_internal[5];
        let d_4 = x_rel_2 * d_5 /*-0*/+ self.coeffs_internal[4];
        let d_3 = x_rel_2 * d_4 - d_5 + self.coeffs_internal[3];
        let d_2 = x_rel_2 * d_3 - d_4 + self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 - d_3 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_perfect_fit() {
        let approx = ChebyshevExpansion::<15>::fit(0.0, 1.0, |x| 1.0 + 2.0 * x + 3.0 * x * x);
        let eps = 1e-6;
        for (i, c) in approx.coeffs().iter().enumerate() {
            if i > 2 {
                assert!(c.abs() < eps)
            } else {
                assert!(c.abs() > 0.3);
            }
        }
        assert!((approx.eval(0.0) - 1.0).abs() < eps);
        assert!((approx.eval(1.0) - 6.0).abs() < eps);
    }

    #[test]
    fn test_is_even() {
        let f = |x: f32| x.cos();
        let eps = 1e-6;
        let approx_even = ChebyshevExpansion::<15>::fit(-1.5, 1.5, f);
        assert!(approx_even.is_even(eps));
        let approx_not_even = ChebyshevExpansion::<15>::fit(-1.0, 2.0, f);
        assert!(!approx_not_even.is_even(eps));
    }

    #[test]
    fn test_is_odd() {
        let f = |x: f32| x.sin();
        let eps = 1e-6;
        let approx_odd = ChebyshevExpansion::<15>::fit(-1.5, 1.5, f);
        assert!(approx_odd.is_odd(eps));
        let approx_not_odd = ChebyshevExpansion::<15>::fit(-1.0, 2.0, f);
        assert!(!approx_not_odd.is_odd(eps));
    }

    #[test]
    fn test_eval_trunc() {
        let x_min = 0.1;
        let x_max = 1.2;
        let x = 0.5 * (x_min + x_max);
        let f = |x: f32| x.sqrt();
        let approx_10 = ChebyshevExpansion::<10>::fit(x_min, x_max, f);
        let approx_3 = ChebyshevExpansion::new(
            x_min,
            x_max,
            [
                approx_10.coeffs()[0],
                approx_10.coeffs()[1],
                approx_10.coeffs()[2],
            ],
        );
        assert!(approx_3.eval(x) == approx_10.eval_trunc(x, 3));
    }

    #[test]
    fn test_loop_free_eval() {
        let x_min = 0.1;
        let x_max = 2.0;
        let x = 0.5 * (x_max + x_min);
        let f = |x: f32| x.sqrt();
        let approx_2 = ChebyshevExpansion::<2>::fit(x_min, x_max, f);
        assert!(approx_2.eval(x) == approx_2.eval_2(x));
        let approx_3 = ChebyshevExpansion::<3>::fit(x_min, x_max, f);
        assert!(approx_3.eval(x) == approx_3.eval_3(x));
        let approx_4 = ChebyshevExpansion::<4>::fit(x_min, x_max, f);
        assert!(approx_4.eval(x) == approx_4.eval_4(x));
        let approx_5 = ChebyshevExpansion::<5>::fit(x_min, x_max, f);
        assert!(approx_5.eval(x) == approx_5.eval_5(x));
        let approx_6 = ChebyshevExpansion::<6>::fit(x_min, x_max, f);
        assert!(approx_6.eval(x) == approx_6.eval_6(x));
    }

    #[test]
    fn test_loop_free_eval_neg1_to_1() {
        let x_min = -1.0;
        let x_max = 1.0;
        let x = 0.893;
        let f = |x: f32| x.cos();
        let approx_2 = ChebyshevExpansion::<2>::fit(x_min, x_max, f);
        assert!(approx_2.eval(x) == approx_2.eval_2(x));
        assert!(approx_2.eval(x) == approx_2.eval_2_neg1_to_1(x));
        let approx_3 = ChebyshevExpansion::<3>::fit(x_min, x_max, f);
        assert!(approx_3.eval(x) == approx_3.eval_3(x));
        assert!(approx_3.eval(x) == approx_3.eval_3_neg1_to_1(x));
        let approx_4 = ChebyshevExpansion::<4>::fit(x_min, x_max, f);
        assert!(approx_4.eval(x) == approx_4.eval_4(x));
        assert!(approx_4.eval(x) == approx_4.eval_4_neg1_to_1(x));
        let approx_5 = ChebyshevExpansion::<5>::fit(x_min, x_max, f);
        assert!(approx_5.eval(x) == approx_5.eval_5(x));
        assert!(approx_5.eval(x) == approx_5.eval_5_neg1_to_1(x));
        let approx_6 = ChebyshevExpansion::<6>::fit(x_min, x_max, f);
        assert!(approx_6.eval(x) == approx_6.eval_6(x));
        assert!(approx_6.eval(x) == approx_6.eval_6_neg1_to_1(x));
    }

    #[test]
    fn test_boundary_matching() {
        let x_min = 0.1;
        let x_max = 2.0;
        let f = |x: f32| x.sqrt();
        let approx_match_none = ChebyshevExpansion::<3>::fit_with_options(
            x_min,
            x_max,
            f,
            |x| x.cos(),
            MatchBoundary::None,
        );
        let approx_x_min = approx_match_none.eval(x_min);
        let approx_x_max = approx_match_none.eval(x_max);
        let f_x_min = f(x_min);
        let f_x_max = f(x_max);
        let eps = 1e-7;

        let approx_match_both = ChebyshevExpansion::<3>::fit_with_options(
            x_min,
            x_max,
            f,
            |x| x.cos(),
            MatchBoundary::Both,
        );
        assert!((approx_match_both.eval(x_min) - f_x_min).abs() < eps);
        assert!((approx_match_both.eval(x_max) - f_x_max).abs() < eps);

        let approx_match_left = ChebyshevExpansion::<3>::fit_with_options(
            x_min,
            x_max,
            f,
            |x| x.cos(),
            MatchBoundary::Min,
        );
        assert!((approx_match_left.eval(x_min) - f_x_min).abs() < eps);
        assert!((approx_match_left.eval(x_max) - approx_x_max).abs() < eps);

        let approx_match_right = ChebyshevExpansion::<3>::fit_with_options(
            x_min,
            x_max,
            f,
            |x| x.cos(),
            MatchBoundary::Max,
        );
        assert!((approx_match_right.eval(x_min) - approx_x_min).abs() < eps);
        assert!((approx_match_right.eval(x_max) - f_x_max).abs() < eps);
    }
}
