use std::iter::Sum;

use num_traits::{Num, ToPrimitive};

use crate::stats::{Column, ColumnError, EDAError};

pub fn min<T: Num + Copy + PartialOrd>(data: &[T]) -> Result<T, EDAError> {
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

pub fn max<T: Num + Copy + PartialOrd>(data: &[T]) -> Result<T, EDAError> {
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

pub fn range<T: Num + Copy + PartialOrd>(data: &[T]) -> Result<T, EDAError> {
    let range = max(data)? - min(data)?;
    Ok(range)
}

// NOTE: this (and everything using these conversions) should pretty much never fail,
// so unwrapping that conversion is okay!!1!
pub fn mean<T: Num + ToPrimitive + Copy + Sum>(data: &[T]) -> Result<f64, EDAError> {
    let n = data.len();

    if n == 0 {
        return Err(EDAError::EmptyData);
    }

    let sum = data.iter().copied().sum::<T>().to_f64().unwrap();
    Ok(sum / n as f64)
}

pub fn weighted_mean<T, U>(data: &[T], weights: &[U]) -> Result<f64, EDAError>
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

pub fn median<T: Num + ToPrimitive + Copy + Sum>(data: &[T]) -> Result<f64, EDAError> {
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
}
