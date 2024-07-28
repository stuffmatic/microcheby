use microcheby::ChebyshevExpansion;

const SQRT_APPROX: ChebyshevExpansion<6> = ChebyshevExpansion::const_new(
    0.1,
    4.0 / (1.0 - 0.1),
    [
        0.7033246,
        0.32991815,
        -0.04125017,
        0.010474294,
        -0.0032901317,
        0.0010244437,
    ],
);

fn sqrt_approx() {
    // Compute an approximation of the square root of x on the interval [0.1, 1.0]
    let sqrt_approx = ChebyshevExpansion::<6>::fit(0.1, 1.0, |x| x.sqrt());
    println!("{sqrt_approx}");
    println!("{:?}", sqrt_approx);
    // Get the approximated value at x = 0.7
    let value_approx = sqrt_approx.eval(0.7);
    // Get the actual value at x = 0.7
    let value_actual = 0.7_f32.sqrt();
    // Compute the approximation error at x = 0.7 and make sure it's small
    let error = value_actual - value_approx;
    assert!(error.abs() < 0.0002);
}

fn main() {
    sqrt_approx();
}
