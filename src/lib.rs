#![cfg_attr(not(feature = "std"), no_std)]

//! microcheby is a crate for computing and evaluating polynomial approximations of
//! one dimensional functions using [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials). 
//! The basic idea is that a function can be expressed as an infinite weighted
//! sum of polynomials of increasing order, a so called Chebyshev expansion. If the function is sufficiently well behaved, the coefficients (weights) will 
//! typically converge to zero quickly and only the first few terms are needed to get a good approximation.
//! For a truncated expansion with _n_ terms, a rough estimate of the approximation error is given by
//! the magnitude of coefficient _n+1_.
//! 
//! The code is `no_std` compatible and optimized for resource
//! constrained environments where every clock cycle counts. Optimizations include:
//! 
//! * [Clenshaw recursion](https://en.wikipedia.org/wiki/Clenshaw_algorithm) for evaluating approximations.
//! * Efficient loop free functions for evaluating low order approximations.
//! * Even more efficient loop free evaluation if the range happens to be [-1, 1].
//!
//! # Basic usage
//! 
//! ```
//! use microcheby::ChebyshevApprox;
//! 
//! // Compute a 6 term approximation of the square root of x on the interval [0.1, 1.0]
//! let sqrt_approx = ChebyshevApprox::<6>::fit(0.1, 1.0, |x| x.sqrt());
//! // Get the approximated value at x=0.7
//! let value_approx = sqrt_approx.eval(0.7);
//! // Get the actual value at x=0.7
//! let value_actual = 0.7_f32.sqrt();
//! // Compute the approximation error
//! let error = value_actual - value_approx;
//! assert!(error.abs() < 0.0002);
//! ```
//! 
//! # Precomputing and instantiating approximations
//! 
//! Computing approximations requires an accurate cosine function and potentially costly function evaluations.
//! Also, it can't be done in constant expressions at compile time. This means it is sometimes desirable to precompute 
//! approximations and then instantiate them in your code.
//! 
//! ```
//! use microcheby::ChebyshevApprox;
//! 
//! // Compute an approximation of the square root of x on the interval [0.1, 1.0]
//! let sqrt_approx = ChebyshevApprox::<6>::fit(0.1, 1.0, |x| x.sqrt());
//! 
//! // x_min, x_max and coeffs are needed to instantiate the approximation
//! // You can either print them to the terminal like this or use the associated getters.
//! println!("{sqrt_approx}");
//! 
//! // Instantiate the approximation computed above using arguments copied from the terminal.
//! let approx = ChebyshevApprox::new(
//!     0.1, 
//!     1.0, 
//!     [1.4066492, 0.32991815, -0.04125017, 0.010474294, -0.0032901317, 0.0010244437]
//! );
//! assert_eq!(sqrt_approx.x_min(), approx.x_min());
//! assert_eq!(sqrt_approx.x_max(), approx.x_max());
//! assert_eq!(sqrt_approx.coeffs(), approx.coeffs());
//! ```
//! 
//! Approximations can also be instantiated statically at compile time:
//! 
//! ```
//! use microcheby::ChebyshevApprox;
//! 
//! // Compute an approximation of the square root of x on the interval [0.1, 1.0]
//! let sqrt_approx = ChebyshevApprox::<6>::fit(0.1, 1.0, |x| x.sqrt());
//! 
//! // x_min, x_scale and coeffs_internal are needed to instantiate the approximation
//! // at compile time since floating point operations are not allowed in constant expressions.
//! // You can either print them to the terminal like this or get them using a debugger.
//! println!("{:?}", sqrt_approx);
//! 
//! const APPROX:ChebyshevApprox<6> = ChebyshevApprox::const_new(
//!     0.1, 
//!     4.4444447, 
//!     [0.7033246, 0.32991815, -0.04125017, 0.010474294, -0.0032901317, 0.0010244437]
//! );
//! assert_eq!(sqrt_approx.x_min(), APPROX.x_min());
//! assert_eq!(sqrt_approx.x_max(), APPROX.x_max());
//! assert_eq!(sqrt_approx.coeffs(), APPROX.coeffs());
//! ```

use core::f32::consts::PI;

#[cfg(feature = "std")]
use core::fmt;

/// Boundary matching is the process of adding a linear
/// term `ax + b` to the Chebyshev expansion to make the resulting approximation
/// match the original function exactly at `x_min` and/or `x_max`. 
/// Obviously, this increases the approximation error, but 
/// can be useful when exact matches at `x_min` and/or `x_max` are
/// more important than overall accuracy.
#[derive(PartialEq)]
pub enum BoundaryMatching {
    None,
    Min,
    Max,
    Both,
}

/// An approximation of a TODO univariate? one dimensional function as a linear combination
/// of the N first Chebyshev polynomials.
/// See [Numerical Recipes in C: The Art of Scientific Computing](http://www.ff.bg.ac.rs/Katedre/Nuklearna/SiteNuklearna/bookcpdf/c5-8.pdf).
#[derive(Clone, Copy)]
pub struct ChebyshevApprox<const N: usize> {
    /// Chebyshev polynomial coefficients. NOTE: Coefficient [0] TODO
    coeffs_internal: [f32; N],
    x_min: f32,
    /// 4.0 / (x_max - x_min)
    range_scale: f32,
}

#[cfg(feature = "std")]
impl<const N: usize> fmt::Display for ChebyshevApprox<N> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        writeln!(fmt, "ChebyshevApprox {{")?;
        writeln!(fmt, "  x_min: {},", self.x_min())?;
        writeln!(fmt, "  x_max: {},",self.x_max())?;
        writeln!(fmt, "  coeffs: {:?}", self.coeffs())?;
        writeln!(fmt, "}}")?;
        Ok(())
    }
}

#[cfg(feature = "std")]
impl<const N: usize> fmt::Debug for ChebyshevApprox<N> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        writeln!(fmt, "ChebyshevApprox {{")?;
        writeln!(fmt, "  x_min: {},", self.x_min())?;
        writeln!(fmt, "  range_scale: {},",self.range_scale)?;
        writeln!(fmt, "  coeffs_internal: {:?}", self.coeffs_internal)?;
        writeln!(fmt, "}}")?;
        Ok(())
    }
}

impl<const N: usize> ChebyshevApprox<N> {
    /// Creates a `ChebyshevApprox` instance approximating a given function on a given range.
    /// 
    /// # Arguments
    /// * `x_min` The start of the range to approximate. 
    /// * `x_max` The end of the range to approximate.
    /// * `f` The function to approximate 
    #[cfg(feature = "std")]
    pub fn fit<F>(x_min: f32, x_max: f32, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        ChebyshevApprox::fit_with_options(x_min, x_max, |x: f32| x.cos(), f, BoundaryMatching::None)
    }

    /// Creates a `ChebyshevApprox` instance approximating a given function on a given range,
    /// optionally performing boundary matching. 
    /// 
    /// # Arguments
    /// * `x_min` The start of the range to approximate. 
    /// * `x_max` The end of the range to approximate.
    /// * `cos` A function for computing the cosine of x. Useful in `no_std` environments 
    /// to provide a custom cosine implementation since the standard math functions are not available.
    /// * `f` The function to approximate 
    /// * `match_boundary` 
    pub fn fit_with_options<F, G>(
        x_min: f32,
        x_max: f32,
        cos: G,
        f: F,
        match_boundary: BoundaryMatching,
    ) -> Self
    where
        F: Fn(f32) -> f32,
        G: Fn(f32) -> f32,
    {
        let mut approx = ChebyshevApprox::new(x_min, x_max, [0.0; N]);
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

        if match_boundary != BoundaryMatching::None && N > 1 {
            // Add a linear term a + bx that offsets the left and right
            // ends to the desired values
            let (x_min_offs, x_max_offs) = match match_boundary {
                BoundaryMatching::Min => (f(x_min) - approx.eval(x_min), 0.0),
                BoundaryMatching::Max => (0.0, f(x_max) - approx.eval(x_max)),
                BoundaryMatching::Both => {
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

    /// Create a `ChebyshevApprox` instance using a given range and 
    /// coefficients.
    pub fn new(x_min: f32, x_max: f32, coeffs: [f32; N]) -> Self {
        let mut coeffs_internal = coeffs;
        coeffs_internal[0] *= 0.5;
        ChebyshevApprox::const_new(
            x_min, 
            4.0 / (x_max - x_min), 
            coeffs_internal
        )
    }

    /// Creates a `ChebyshevApprox` instance statically. Due to floating point 
    /// limitations in const expressions, `range_scale` and `coeffs_internal` must be passed "raw".
    /// The values for these can be printed using `println!("{approx:?}")` and then copy-pasted
    /// (`approx` is a `ChebyshevApprox` instance).
    /// 
    /// # Arguments
    /// 
    /// * `x_min` The start of the approximated function's range.
    /// * `range_scale` Equal to `4.0 / (x_max - x_min)`, where `x_max` is the end of the approximated function's range.
    /// * `coeffs_internal` The same as the Chebychev weights, except that the element at index 0 has been divided by 2.
    pub const fn const_new(x_min: f32, range_scale: f32, coeffs_internal: [f32; N]) -> Self {
        ChebyshevApprox {
            x_min,
            coeffs_internal,
            range_scale,
        }
    }

    pub fn x_min(&self) -> f32 {
        self.x_min
    }

    pub fn x_max(&self) -> f32 {
        4.0 / self.range_scale + self.x_min
    }

    pub fn coeffs(&self) -> [f32; N] {
        let mut coeffs = self.coeffs_internal;
        if N > 0 {
            // compensate for 'pre baked' multiplication by 0.5
            coeffs[0] *= 2.0;
        }
        coeffs
    }

    pub fn polynomial_coeffs(&self) -> [f32; N] {
        let mut tn_2 = [0.0; N];
        let mut tn_1 = [0.0; N];
        if N > 0 {
            tn_2[0] = 1.0;
        }
        if N > 1 {
            tn_1[1] = 1.0;
        }

        let mut result = [0.0; N];
        for i in 0..N {
            let tn = match i {
                0 => tn_2,
                1 => tn_1,
                _ => {
                    let mut tn_next = [0.0; N];
                    for j in 0..N {
                        tn_next[j] -= tn_2[j];
                        if j < N - 1 {
                            tn_next[j + 1] += 2.0 * tn_1[j];
                        }
                    }
                    tn_next
                }
            };

            let ci = if i == 0 {
                // compensate for 'pre baked' multiplication by 0.5
                2.0 * self.coeffs_internal[i]
            } else {
                self.coeffs_internal[i]
            };

            for j in 0..N {
                result[j] += ci * tn[j];
            }
            if i == 0 {
                result[0] -= 0.5 * ci;
            }
            if i > 1 {
                tn_2 = tn_1;
                tn_1 = tn;
            }
        }
        let x_min = self.x_min;
        let x_max = 4.0 / self.range_scale + self.x_min;
        let a = 0.5 * (x_min + x_max);
        let b = 0.5 * (x_max - x_min);
        let mut mapped_result = [0.0; N];
        for i in 0..N {
            // substitute x in x^i by (ax + b)
            for j in 0..=i {
                // https://en.wikipedia.org/wiki/Binomial_coefficient
                let bin_coeff = if j == 0 {
                    1.0
                } else {
                    let mut den = 1;
                    for k in 1..=j {
                        den *= k;
                    }
                    ((i * (i - 1)) as f32) / (den as f32)
                };
                // mapped_result[i - 1] +=
            }
        }
        mapped_result
    }

    /// Evaluate the using Clenshaw recursion.
    /// 
    /// # Arguments
    /// * `x` - Evaluate the approximation at this x value.
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

    /// Evaluate using Clenshaw recursion, assuming the range of the approximation is [-1, 1].
    /// 
    /// # Arguments
    /// * `x` - Evaluate the approximation at this x value. Assumed to be in the range [-1, 1].
    pub fn eval_neg1_to_1(&self, x: f32) -> f32 {
        let x_rel_2 = 2.0 * x;
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

    /// Evaluate up to a given order using Clenshaw recursion
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

    pub fn is_odd(&self, eps: f32) -> bool {
        for ci in self.coeffs_internal.iter().step_by(2) {
            if ci.abs() > eps {
                return false;
            }
        }

        true
    }

    pub fn is_even(&self, eps: f32) -> bool {
        for (i, ci) in self.coeffs_internal.iter().skip(1).step_by(2).enumerate() {
            let scale = if i == 0 { 2.0 } else { 1.0 }; // account for "baked" c0 scale of 0.5
            if scale * ci.abs() > eps {
                return false;
            }
        }

        true
    }
}

impl ChebyshevApprox<2> {
    /// Optimized order 2 Clenshaw recursion with no loops.
    pub fn eval_2(&self, x: f32) -> f32 {
        let x_rel_2 = -2.0 + (x - self.x_min) * self.range_scale;
        0.5 * x_rel_2 * self.coeffs_internal[1] + self.coeffs_internal[0]
    }

    pub fn eval_2_neg1_to_1(&self, x: f32) -> f32 {
        let x_rel_2 = 2.0 * x;
        0.5 * x_rel_2 * self.coeffs_internal[1] + self.coeffs_internal[0]
    }
}

impl ChebyshevApprox<3> {
    /// Optimized order 3 Clenshaw recursion with no loops.
    pub fn eval_3(&self, x: f32) -> f32 {
        let x_rel_2 = -2.0 + (x - self.x_min) * self.range_scale;
        let d_2 = self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }

    pub fn eval_3_neg1_to_1(self, x: f32) -> f32 {
        let x_rel_2 = 2.0 * x;
        let d_2 = self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }
}

impl ChebyshevApprox<4> {
    /// Optimized order 4 Clenshaw recursion with no loops.
    pub fn eval_4(&self, x: f32) -> f32 {
        let x_rel_2 = -2.0 + (x - self.x_min) * self.range_scale;
        let d_3 = self.coeffs_internal[3];
        let d_2 = x_rel_2 * d_3 + self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 - d_3 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }

    pub fn eval_4_neg1_to_1(&self, x: f32) -> f32 {
        let x_rel_2 = 2.0 * x;
        let d_3 = self.coeffs_internal[3];
        let d_2 = x_rel_2 * d_3 + self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 - d_3 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }
}

impl ChebyshevApprox<5> {
    /// Optimized order 5 Clenshaw recursion with no loops.
    pub fn eval_5(&self, x: f32) -> f32 {
        let x_rel_2 = -2.0 + (x - self.x_min) * self.range_scale;
        let d_4 = self.coeffs_internal[4];
        let d_3 = x_rel_2 * d_4 /*-0*/+ self.coeffs_internal[3];
        let d_2 = x_rel_2 * d_3 - d_4 + self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 - d_3 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }

    pub fn eval_5_neg1_to_1(&self, x: f32) -> f32 {
        let x_rel_2 = 2.0 * x;
        let d_4 = self.coeffs_internal[4];
        let d_3 = x_rel_2 * d_4 /*-0*/+ self.coeffs_internal[3];
        let d_2 = x_rel_2 * d_3 - d_4 + self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 - d_3 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }
}

impl ChebyshevApprox<6> {
    /// Optimized order 6 Clenshaw recursion with no loops.
    pub fn eval_6(&self, x: f32) -> f32 {
        let x_rel_2 = -2.0 + (x - self.x_min) * self.range_scale;
        let d_5 = self.coeffs_internal[5];
        let d_4 = x_rel_2 * d_5 /*-0*/+ self.coeffs_internal[4];
        let d_3 = x_rel_2 * d_4 - d_5 + self.coeffs_internal[3];
        let d_2 = x_rel_2 * d_3 - d_4 + self.coeffs_internal[2];
        let d_1 = x_rel_2 * d_2 - d_3 + self.coeffs_internal[1];
        0.5 * x_rel_2 * d_1 - d_2 + self.coeffs_internal[0]
    }

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
    fn test_polynomial_coeffs() {
        let approx =
            ChebyshevApprox::<5>::fit(-1.0, 1.0, |x| 1.0 + 2.0 * x + 3.0 * x * x + 4.0 * x * x * x);
        let c = approx.polynomial_coeffs();
        let a = 0;
    }

    #[test]
    fn test_quadratic_perfect_fit() {
        let approx = ChebyshevApprox::<15>::fit(0.0, 1.0, |x| 1.0 + 2.0 * x + 3.0 * x * x);
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
        let approx_even = ChebyshevApprox::<15>::fit(-1.5, 1.5, f);
        assert!(approx_even.is_even(eps));
        let approx_not_even = ChebyshevApprox::<15>::fit(-1.0, 2.0, f);
        assert!(!approx_not_even.is_even(eps));
    }

    #[test]
    fn test_is_odd() {
        let f = |x: f32| x.sin();
        let eps = 1e-6;
        let approx_odd = ChebyshevApprox::<15>::fit(-1.5, 1.5, f);
        assert!(approx_odd.is_odd(eps));
        let approx_not_odd = ChebyshevApprox::<15>::fit(-1.0, 2.0, f);
        assert!(!approx_not_odd.is_odd(eps));
    }

    #[test]
    fn test_eval_neg1_to_1() {
        let x_min = -1.0;
        let x_max = 1.0;
        let x = 0.5 * (x_min + x_max);
        let f = |x: f32| x.sin();
        let approx = ChebyshevApprox::<10>::fit(x_min, x_max, f);
        assert!(approx.eval(x) == approx.eval_neg1_to_1(x));
        let approx_2 = ChebyshevApprox::<10>::fit(-0.9, x_max, f);
        assert!(approx_2.eval(x) != approx_2.eval_neg1_to_1(x));
    }

    #[test]
    fn test_eval_trunc() {
        let x_min = 0.1;
        let x_max = 1.2;
        let x = 0.5 * (x_min + x_max);
        let f = |x: f32| x.sqrt();
        let approx_10 = ChebyshevApprox::<10>::fit(x_min, x_max, f);
        let approx_3 = ChebyshevApprox::new(
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
        let approx_2 = ChebyshevApprox::<2>::fit(x_min, x_max, f);
        assert!(approx_2.eval(x) == approx_2.eval_2(x));
        let approx_3 = ChebyshevApprox::<3>::fit(x_min, x_max, f);
        assert!(approx_3.eval(x) == approx_3.eval_3(x));
        let approx_4 = ChebyshevApprox::<4>::fit(x_min, x_max, f);
        assert!(approx_4.eval(x) == approx_4.eval_4(x));
        let approx_5 = ChebyshevApprox::<5>::fit(x_min, x_max, f);
        assert!(approx_5.eval(x) == approx_5.eval_5(x));
        let approx_6 = ChebyshevApprox::<6>::fit(x_min, x_max, f);
        assert!(approx_6.eval(x) == approx_6.eval_6(x));
    }

    #[test]
    fn test_boundary_matching() {
        let x_min = 0.1;
        let x_max = 2.0;
        let f = |x: f32| x.sqrt();
        let approx_match_none = ChebyshevApprox::<3>::fit_with_options(
            x_min,
            x_max,
            |x| x.cos(),
            f,
            BoundaryMatching::None,
        );
        let approx_x_min = approx_match_none.eval(x_min);
        let approx_x_max = approx_match_none.eval(x_max);
        let f_x_min = f(x_min);
        let f_x_max = f(x_max);
        let eps = 1e-7;

        let approx_match_both = ChebyshevApprox::<3>::fit_with_options(
            x_min,
            x_max,
            |x| x.cos(),
            f,
            BoundaryMatching::Both,
        );
        assert!((approx_match_both.eval(x_min) - f_x_min).abs() < eps);
        assert!((approx_match_both.eval(x_max) - f_x_max).abs() < eps);

        let approx_match_left = ChebyshevApprox::<3>::fit_with_options(
            x_min,
            x_max,
            |x| x.cos(),
            f,
            BoundaryMatching::Min,
        );
        assert!((approx_match_left.eval(x_min) - f_x_min).abs() < eps);
        assert!((approx_match_left.eval(x_max) - approx_x_max).abs() < eps);

        let approx_match_right = ChebyshevApprox::<3>::fit_with_options(
            x_min,
            x_max,
            |x| x.cos(),
            f,
            BoundaryMatching::Max,
        );
        assert!((approx_match_right.eval(x_min) - approx_x_min).abs() < eps);
        assert!((approx_match_right.eval(x_max) - f_x_max).abs() < eps);
    }
}
