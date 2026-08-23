use std::iter::Sum;

use num_traits::{AsPrimitive, Num, ToPrimitive};

use crate::stats::{ColumnError, mean};

// NOTE: this can be done single-pass instead of computing everything one by one, should be optimized eventually
pub fn r_corr<T, U>(a: &[T], b: &[U]) -> Result<f64, ColumnError>
where
    T: ToPrimitive + Sum + AsPrimitive<f64>,
    U: ToPrimitive + Sum + AsPrimitive<f64>,
{
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
        .map(|(&a0, &b0)| a0.as_() * b0.as_())
        .sum::<f64>();
    let sum_a = a.iter().copied().sum::<T>().to_f64().unwrap(); // use "copied()" to turn &T into T
    let sum_b = b.iter().copied().sum::<U>().to_f64().unwrap();
    let sum_a_squared = a.iter().map(|&x| x.as_() * x.as_()).sum::<f64>();
    let sum_b_squared = b.iter().map(|&x| x.as_() * x.as_()).sum::<f64>();

    // fraction calculation
    let numer = n * sum_ab - sum_a * sum_b;
    let denom = ((n * sum_a_squared - sum_a * sum_a) * (n * sum_b_squared - sum_b * sum_b)).sqrt();

    // full fraction
    let r = numer / denom;
    Ok(r)
}

// the shared code between population and sample covariance
// NOTE: this can probably be done single pass too
fn cov_no_denominator<T, U>(a: &[T], b: &[U]) -> Result<f64, ColumnError>
where
    T: Num + ToPrimitive + Sum + AsPrimitive<f64>,
    U: Num + ToPrimitive + Sum + AsPrimitive<f64>,
{
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

pub fn pop_cov<T, U>(a: &[T], b: &[U]) -> Result<f64, ColumnError>
where
    T: Num + ToPrimitive + Sum + AsPrimitive<f64>,
    U: Num + ToPrimitive + Sum + AsPrimitive<f64>,
{
    let cov = cov_no_denominator(a, b)?;
    Ok(cov / a.len() as f64)
}

pub fn samp_cov<T, U>(a: &[T], b: &[U]) -> Result<f64, ColumnError>
where
    T: Num + ToPrimitive + Sum + AsPrimitive<f64>,
    U: Num + ToPrimitive + Sum + AsPrimitive<f64>,
{
    let cov = cov_no_denominator(a, b)?;
    Ok(cov / (a.len() - 1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let cov1 = pop_cov::<i32, i32>(&a, &b).unwrap();
        assert_eq!(cov1, 2.5);

        // test sample covariance
        let cov2 = samp_cov::<i32, i32>(&a, &b).unwrap();
        assert_eq!(cov2, 10. / 3.);
    }
}
