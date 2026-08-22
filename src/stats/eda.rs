use std::iter::Sum;

use crate::stats::column::ColumnError;
use num_traits::{AsPrimitive, Num, ToPrimitive};

// general methods for slices

pub fn min<T: Num + Copy + PartialOrd>(data: &[T]) -> Result<T, ColumnError> {
    if data.len() == 0 {
        return Err(ColumnError::Empty);
    }

    let mut min: T = data[0];
    for x in &data[1..] {
        if *x < min {
            min = *x;
        }
    }

    Ok(min)
}

pub fn max<T: Num + Copy + PartialOrd>(data: &[T]) -> Result<T, ColumnError> {
    if data.len() == 0 {
        return Err(ColumnError::Empty);
    }

    let mut max: T = data[0];
    for x in &data[1..] {
        if *x > max {
            max = *x;
        }
    }

    Ok(max)
}

// NOTE: this (and everything below) should pretty much never fail,
// so unwrapping that conversion is okay!!1!
pub fn mean<T: Num + ToPrimitive + Copy + Sum>(data: &[T]) -> Result<f64, ColumnError> {
    let n = data.len();

    if n == 0 {
        return Err(ColumnError::Empty);
    }

    let sum = data.iter().copied().sum::<T>().to_f64().unwrap();
    Ok(sum / n as f64)
}

pub fn median<T: Num + ToPrimitive + Copy + Sum>(data: &[T]) -> Result<f64, ColumnError> {
    let med: f64;
    let n = data.len();

    if n == 0 {
        return Err(ColumnError::Empty);
    }

    if n % 2 == 0 {
        med = (data[n / 2 - 1].to_f64().unwrap() + data[n / 2].to_f64().unwrap()) / 2.;
    } else {
        med = data[n / 2].to_f64().unwrap();
    }

    Ok(med)
}

pub fn linear_quantile<T: Num + ToPrimitive>(data: &[T], q: f64) -> Result<f64, ColumnError> {
    // linear interpolation has the formula:
    //   x_floor(q) + (q - floor(q)) (x_ceil(q) - x_floor(q))
    // where q is the quantile

    let n: usize = data.len();

    if n == 0 {
        return Err(ColumnError::Empty);
    }

    let h: f64 = q * ((n - 1) as f64);

    let i: usize = h.floor() as usize;
    let j: usize = h.ceil() as usize;
    let d: f64 = h - i as f64;

    let x_i: f64 = data[i].to_f64().unwrap();
    let x_j: f64 = data[j].to_f64().unwrap();

    let x_diff = x_j - x_i;
    Ok(x_i + (d * x_diff))
}

pub fn nearest_quantile<T: AsPrimitive<f64>>(data: &[T], q: f64) -> Result<f64, ColumnError> {
    let n = data.len();

    if n == 0 {
        return Err(ColumnError::Empty);
    }

    let idx = ((n - 1) as f64 * q).round(); // the index at q% of the array's length
    Ok(data[idx as usize].as_())
}

// NOTE: this can be done single-pass, should be optimized eventually
pub fn r_corr<T: Num + ToPrimitive + Copy + Sum>(a: &[T], b: &[T]) -> Result<f64, ColumnError> {
    if a.len() != b.len() {
        return Err(ColumnError::NonMatchingSizes);
    } else if a.len() == 0 {
        return Err(ColumnError::Empty);
    }

    let n = a.len() as f64;

    // sum computations (abomination of types)
    let sum_ab = a
        .iter()
        .zip(b.iter())
        .map(|(a0, b0)| *a0 * *b0)
        .sum::<T>()
        .to_f64()
        .unwrap();
    let sum_a = a.iter().copied().sum::<T>().to_f64().unwrap(); // use "copied()" to turn &T into T
    let sum_b = b.iter().copied().sum::<T>().to_f64().unwrap();
    let sum_a_squared = a.iter().map(|&x| x * x).sum::<T>().to_f64().unwrap();
    let sum_b_squared = b.iter().map(|&x| x * x).sum::<T>().to_f64().unwrap();

    // fraction calculation
    let numer = n * sum_ab - sum_a * sum_b;
    let denom = ((n * sum_a_squared - sum_a * sum_a) * (n * sum_b_squared - sum_b * sum_b)).sqrt();

    // full fraction
    let r = numer / denom;
    Ok(r)
}

// the shared code between population and sample covariance
// NOTE: this can probably be done single pass too
fn cov_no_denominator<T: Num + ToPrimitive + Copy + Sum>(
    a: &[T],
    b: &[T],
) -> Result<f64, ColumnError> {
    if a.len() != b.len() {
        return Err(ColumnError::NonMatchingSizes);
    } else if a.len() == 0 {
        return Err(ColumnError::Empty);
    }

    let mean_a = mean(a)?;
    let mean_b = mean(b)?;

    let mut sum: f64 = 0.;
    a.iter().zip(b).for_each(|(&a0, &b0)| {
        sum += (a0.to_f64().unwrap() - mean_a) * (b0.to_f64().unwrap() - mean_b)
    });

    Ok(sum as f64)
}

pub fn pop_cov<T: Num + ToPrimitive + Copy + Sum>(a: &[T], b: &[T]) -> Result<f64, ColumnError> {
    let cov = cov_no_denominator(a, b)?;
    Ok(cov / a.len() as f64)
}

pub fn samp_cov<T: Num + ToPrimitive + Copy + Sum>(a: &[T], b: &[T]) -> Result<f64, ColumnError> {
    let cov = cov_no_denominator(a, b)?;
    Ok(cov / (a.len() - 1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_equals_median() {
        // (1+2+3+4+5)/5 = 3 (mean)
        // and
        // middle is 3 (median)
        // so the median and mean should be equal theoretically

        let arr = [1, 2, 3, 4, 5];

        // mean test
        let m = mean(&arr).unwrap();
        assert_eq!(m, 3.);

        // median test
        let m = median(&arr).unwrap();
        assert_eq!(m, 3.);
    }

    #[test]
    fn simple_eda_with_floats() {
        // ts exists cus i wanna see if PartialOrd works

        let arr = [1., 2., 4., 7., 9.];

        let m = mean(&arr).unwrap();
        assert_eq!(m, 4. + 3. / 5.);

        let m = median(&arr).unwrap();
        assert_eq!(m, 4.);
    }

    #[test]
    fn even_count_median() {
        let arr = [1, 2, 3, 4, 5, 6];

        let med = median(&arr).unwrap();
        assert_eq!(med, 3.5);
    }

    #[test]
    fn simple_quartiles() {
        let arr = [1, 2, 3, 4, 5];

        let q2 = nearest_quantile(&arr, 0.5).unwrap();
        assert_eq!(q2, 3.);

        // third quartile test
        let q3 = nearest_quantile(&arr, 0.75).unwrap();
        assert_eq!(q3, 4.)
    }

    #[test]
    fn second_quartile_equals_median() {
        let arr = [6, 7, 8, 9];
        let med = median(&arr).unwrap();
        let q2 = linear_quantile(&arr, 0.5).unwrap();

        assert_eq!(med, q2);
    }

    #[test]
    fn simple_min_and_max() {
        let arr = [1, 2, 3];

        let m = min(&arr).unwrap();
        assert_eq!(m, 1);

        let m = max(&arr).unwrap();
        assert_eq!(m, 3);
    }

    #[test]
    fn perfect_correlation() {
        let a = [1, 2, 3];
        let b = [2, 4, 6];

        let r = r_corr(&a, &b).unwrap();
        assert_eq!(r, 1.)
    }

    #[test]
    fn simple_covariances() {
        let a = [1, 2, 3, 4];
        let b = [2, 4, 6, 8];

        // test population covariance
        let cov1 = pop_cov(&a, &b).unwrap();
        assert_eq!(cov1, 2.5);

        // test sample covariance
        let cov2 = samp_cov(&a, &b).unwrap();
        assert_eq!(cov2, 10. / 3.);
    }
}
