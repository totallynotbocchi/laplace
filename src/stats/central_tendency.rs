use std::iter::Sum;

use num_traits::{Num, ToPrimitive};

use crate::stats::ColumnError;

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

// NOTE: this (and everything using these conversions) should pretty much never fail,
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
    fn simple_min_and_max() {
        let arr = [1, 2, 3];

        let m = min(&arr).unwrap();
        assert_eq!(m, 1);

        let m = max(&arr).unwrap();
        assert_eq!(m, 3);
    }
}
