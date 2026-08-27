use std::iter::{Product, Sum};

use num_traits::{Num, ToPrimitive};

use crate::stats::{EDAError, EDAResult};

pub fn min<T: Num + Copy + PartialOrd>(data: &[T]) -> EDAResult<T> {
    if data.len() == 0 {
        return Err(EDAError::EmptyData);
    }

    let mut min: T = data[0];
    for x in &data[1..] {
        if *x < min {
            min = *x;
        }
    }

    Ok(min)
}

pub fn max<T: Num + Copy + PartialOrd>(data: &[T]) -> EDAResult<T> {
    if data.len() == 0 {
        return Err(EDAError::EmptyData);
    }

    let mut max: T = data[0];
    for x in &data[1..] {
        if *x > max {
            max = *x;
        }
    }

    Ok(max)
}

pub fn range<T: Num + Copy + PartialOrd>(data: &[T]) -> EDAResult<T> {
    let range = max(data)? - min(data)?;
    Ok(range)
}

// NOTE: this (and everything using these conversions) should pretty much never fail,
// so unwrapping that conversion is okay!!1!
pub fn mean<T: Num + ToPrimitive + Copy + Sum>(data: &[T]) -> EDAResult<f64> {
    let n = data.len();

    if n == 0 {
        return Err(EDAError::EmptyData);
    }

    let sum = data.iter().copied().sum::<T>().to_f64().unwrap();
    Ok(sum / n as f64)
}

pub fn weighted_mean<T, U>(data: &[T], weights: &[U]) -> EDAResult<f64>
where
    T: Num + ToPrimitive + Copy + Sum,
    U: Num + ToPrimitive + Copy,
{
    if data.len() != weights.len() {
        return Err(EDAError::DifferentSizes);
    // one inequality handles either case of being empty
    } else if data.len() == 0 {
        return Err(EDAError::EmptyData);
    }

    let sum = data
        .iter()
        .zip(weights)
        .map(|(&x, &w)| x.to_f64().unwrap() * w.to_f64().unwrap())
        .sum::<f64>();

    Ok(sum / data.len() as f64)
}

pub fn geometric_mean<T: Num + ToPrimitive + Copy + Product>(data: &[T]) -> EDAResult<f64> {
    let prod = data.iter().copied().product::<T>().to_f64().unwrap();
    Ok(prod.powf(1. / data.len() as f64))
}

fn is_valid_percent(p: f64) -> bool {
    p >= 0. && p <= 1.
}

pub fn trimmed_mean<T: Num + ToPrimitive + Copy + Sum>(
    data: &[T],
    left: f64,
    right: f64,
) -> EDAResult<f64> {
    if !is_valid_percent(left) || !is_valid_percent(right) {
        return Err(EDAError::InvalidParameter {
            message: format!(
                "Values have to be valid percentages between 0-1, not {left} and {right}."
            ),
        });
    }

    // cut the data at the first left% and the data at the last right% by ignoring the respective
    // indices (quantiles)
    let start_idx = (left * data.len() as f64) as usize;
    let end_idx = (data.len() as f64 - right * data.len() as f64) as usize;

    let data_slice = &data[start_idx..end_idx];
    let sum = data_slice.iter().copied().sum::<T>().to_f64().unwrap();

    Ok(sum / data_slice.len() as f64)
}

pub fn median<T: Num + ToPrimitive + Copy + Sum>(data: &[T]) -> EDAResult<f64> {
    let med: f64;
    let n = data.len();

    if n == 0 {
        return Err(EDAError::EmptyData);
    }

    if n % 2 == 0 {
        med = (data[n / 2 - 1].to_f64().unwrap() + data[n / 2].to_f64().unwrap()) / 2.;
    } else {
        med = data[n / 2].to_f64().unwrap();
    }

    Ok(med)
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
    fn simple_mean_and_median_with_floats() {
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
    fn simple_min_and_max() {
        let arr = [1, 2, 3];

        let m = min(&arr).unwrap();
        assert_eq!(m, 1);

        let m = max(&arr).unwrap();
        assert_eq!(m, 3);
    }

    #[test]
    fn small_and_large_ranges() {
        // small
        let arr = [1, 6, 100, 193];

        let rn = range(&arr).unwrap();
        assert_eq!(rn, 192);

        // really large and really small
        let arr = [5e-11, 5e-10, 5e-12, 5e12, 5e15, 5e10];
        let rn = range(&arr).unwrap();
        assert_eq!(rn, 5e15 - 5e-12);
    }

    #[test]
    fn weight_mean_by_1_equals_mean() {
        let arr = [1, 2, 3, 7];
        let weights = [1, 1, 1, 1];

        let wm = weighted_mean(&arr, &weights).unwrap();
        let nm = mean(&arr).unwrap();
        assert_eq!(wm, nm);
    }

    #[test]
    fn geometric_mean_correctness() {
        let arr = [1, 3, 6, 12];

        let gm = geometric_mean(&arr).unwrap();
        assert_eq!(gm, 216_f64.powf(1. / 4.));
    }

    #[test]
    fn small_trimmed_mean() {
        let arr = (1..=10).collect::<Vec<i32>>();

        // 10% of 10 is 1, so we remove the min and max
        let tm = trimmed_mean(&arr, 0.1, 0.1).unwrap();
        assert_eq!(tm, 5.5);
    }
}
