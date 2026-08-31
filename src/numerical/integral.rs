use crate::numerical::RealFn;

// left riemann sum formula
fn left_riemann_integral(f: RealFn, a: f64, b: f64, n: usize) -> f64 {
    let dx = (b - a) / n as f64;

    let mut sum = 0.;
    for i in 0..n {
        sum += f(a + i as f64 * dx) * dx;
    }

    sum
}

// right riemann sum formula
fn right_riemann_integral(f: RealFn, a: f64, b: f64, n: usize) -> f64 {
    let dx = (b - a) / n as f64;

    let mut sum = 0.;
    for i in 1..=n {
        sum += f(a + i as f64 * dx) * dx;
    }

    sum
}

// midpoint riemann sum formula
fn mid_riemann_integral(f: RealFn, a: f64, b: f64, n: usize) -> f64 {
    let dx = (b - a) / n as f64;

    let mut sum = 0.;
    for i in 1..=n {
        let mid = (2. * a + i as f64 * dx + (i - 1) as f64 * dx) / 2.;
        sum += f(mid);
    }

    sum * dx
}

// trapezoid rule
fn trapezoid_integral(f: RealFn, a: f64, b: f64, n: usize) -> f64 {
    let dx = (b - a) / n as f64;

    let mut sum = f(a) + f(b);
    for i in 1..n {
        sum += 2. * f(a + i as f64 * dx);
    }

    sum * dx / 2.
}

// simpson's rule
fn even_simpson_integral(f: RealFn, a: f64, b: f64, n: usize) -> f64 {
    let dx = (b - a) / n as f64;

    let mut sum = f(a) + f(b);

    // 1/3 rule
    for i in 1..n {
        let factor = if i % 2 == 0 { 2. } else { 4. };
        sum += factor * f(a + i as f64 * dx);
    }

    sum * dx / 3.
}

#[cfg(test)]
mod tests {
    use super::*;
    static TOLERANCE: f64 = 0.01;

    // this is a RealFn
    // its integral is x^2
    fn f(x: f64) -> f64 {
        2. * x
    }

    #[test]
    fn integral_of_2x() {
        // int[0, 1] 2x dx = x^2 |[0, 1] = 1 - 0 = 1

        let methods = [
            trapezoid_integral,
            mid_riemann_integral,
            right_riemann_integral,
            left_riemann_integral,
            even_simpson_integral,
        ];
        let names = ["trapezoid", "mid", "right", "left", "simpson"];

        for (i, method) in methods.iter().enumerate() {
            print!("{} - ", names[i]);

            let int = method(f, 0., 1., 1000);
            println!("{int}");

            assert!((1. - int).abs() < TOLERANCE)
        }
    }
}
