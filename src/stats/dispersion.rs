use std::iter::Sum;

use num_traits::{AsPrimitive, Num, ToPrimitive};

use crate::stats::{ColumnError, mean};

fn is_valid_quantile(q: f64) -> bool {
    q >= 0. && q <= 1.
}

pub fn linear_quantile<T: Num + ToPrimitive>(data: &[T], q: f64) -> Result<f64, ColumnError> {
    // linear interpolation has the formula:
    //   x_floor(q) + (q - floor(q)) (x_ceil(q) - x_floor(q))
    // where q is the quantile

    let n: usize = data.len();

    if n == 0 {
        return Err(ColumnError::Empty);
    } else if !is_valid_quantile(q) {
        return Err(ColumnError::InvalidInput);
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
    } else if !is_valid_quantile(q) {
        return Err(ColumnError::InvalidInput);
    }

    let idx = ((n - 1) as f64 * q).round(); // the index at q% of the array's length
    Ok(data[idx as usize].as_())
}

pub fn linear_iqr<T: Num + ToPrimitive>(data: &[T], q1: f64, q2: f64) -> Result<f64, ColumnError> {
    if q1 > q2 {
        return Err(ColumnError::InvalidInput);
    }

    let val1 = linear_quantile(data, q1)?;
    let val2 = linear_quantile(data, q2)?;

    Ok(val2 - val1)
}

pub fn nearest_iqr<T: AsPrimitive<f64>>(data: &[T], q1: f64, q2: f64) -> Result<f64, ColumnError> {
    if q1 > q2 {
        return Err(ColumnError::InvalidInput);
    }

    let val1 = nearest_quantile(data, q1)?;
    let val2 = nearest_quantile(data, q2)?;

    Ok(val2 - val1)
}

// the shared code between population and sample standard deviation/variance
fn var_no_denominator<T: Num + ToPrimitive + Copy + Sum>(data: &[T]) -> Result<f64, ColumnError> {
    if data.len() == 0 {
        return Err(ColumnError::Empty);
    }

    let mean = mean(data)?;
    let sum: f64 = data
        .iter()
        .map(|x| (x.to_f64().unwrap() - mean).powi(2))
        .sum::<f64>();

    Ok(sum)
}

pub fn samp_var<T: Num + ToPrimitive + Copy + Sum>(data: &[T]) -> Result<f64, ColumnError> {
    let var = var_no_denominator(data)?;
    Ok(var / (data.len() - 1) as f64)
}

pub fn pop_var<T: Num + ToPrimitive + Copy + Sum>(data: &[T]) -> Result<f64, ColumnError> {
    let var = var_no_denominator(data)?;
    Ok(var / data.len() as f64)
}

pub fn samp_std<T: Num + ToPrimitive + Copy + Sum>(data: &[T]) -> Result<f64, ColumnError> {
    let var = var_no_denominator(data)?;
    Ok(var.sqrt() / ((data.len() - 1) as f64).sqrt())
}

pub fn pop_std<T: Num + ToPrimitive + Copy + Sum>(data: &[T]) -> Result<f64, ColumnError> {
    let var = var_no_denominator(data)?;
    Ok(var.sqrt() / (data.len() as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use crate::stats::median;

    use super::*;

    static TOLERANCE: f64 = 0.000001;

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
    fn zero_variance_and_std() {
        // all values being 0 means variance is 0
        let arr = [2, 2, 2, 2, 2];

        // variance
        let var = pop_var(&arr).unwrap();
        assert_eq!(var, 0.);

        // standard deviation
        let std = pop_std(&arr).unwrap();
        assert_eq!(std, 0.);
    }

    #[test]
    fn simple_vars_and_stds() {
        let arr = [1, 2, 3, 4];

        // test population variance
        let pvar = pop_var(&arr).unwrap();
        assert!((pvar - 5. / 4.).abs() < TOLERANCE);

        // test sample variance
        let svar = samp_var(&arr).unwrap();
        assert!((svar - 5. / 3.).abs() < TOLERANCE);

        // test population standard deviation
        let pdev = pop_std(&arr).unwrap();
        assert!((pdev - 5_f64.sqrt() / 2.).abs() < TOLERANCE);

        // test sample standard deviation
        let sdev = samp_std(&arr).unwrap();
        assert!((sdev - 5_f64.sqrt() / 3_f64.sqrt()).abs() < TOLERANCE);
    }
}
