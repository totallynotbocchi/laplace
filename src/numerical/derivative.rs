use crate::numerical::RealFn;

// central derivative formula
pub fn central_derivative(f: RealFn, x: f64, eps: f64) -> f64 {
    (f(x + eps) - f(x - eps)) / (2. * eps)
}

// forward derivative formula
pub fn forward_derivative(f: RealFn, x: f64, eps: f64) -> f64 {
    (f(x + eps) - f(x)) / eps
}

// backward derivative formula
pub fn backward_derivative(f: RealFn, x: f64, eps: f64) -> f64 {
    (f(x) - f(x - eps)) / eps
}

#[cfg(test)]
mod tests {
    use super::*;
    static TOLERANCE: f64 = 0.0001;

    // this is a RealFn
    // its derivative if 2
    fn f(x: f64) -> f64 {
        2. * x
    }

    #[test]
    fn derivative_of_2x() {
        // d/dx 2x = 2

        let methods = [central_derivative, forward_derivative, backward_derivative];
        let names = ["central", "forward", "backward"];

        for (i, method) in methods.iter().enumerate() {
            print!("{} - ", names[i]);

            let der = method(f, 0., 0.0001);
            println!("{der}");

            assert!((2. - der).abs() < TOLERANCE)
        }
    }
}
