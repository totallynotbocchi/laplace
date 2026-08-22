use crate::stats::column::ColumnError;
use num_traits::{AsPrimitive, Num};

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

pub fn mean<T: AsPrimitive<f64>>(data: &[T]) -> Result<f64, ColumnError> {
    let mut sum: f64 = 0.;
    let n = data.len();

    if n == 0 {
        return Err(ColumnError::Empty);
    }

    data.iter().for_each(|el| sum += el.as_());
    Ok(sum / n as f64)
}

pub fn median<T: AsPrimitive<f64>>(data: &[T]) -> Result<f64, ColumnError> {
    let med: f64;
    let n = data.len();

    if n == 0 {
        return Err(ColumnError::Empty);
    }

    if n % 2 == 0 {
        med = (data[n / 2 - 1].as_() + data[n / 2].as_()) / 2.;
    } else {
        med = data[n / 2].as_();
    }

    Ok(med)
}

pub fn linear_quantile<T: AsPrimitive<f64>>(data: &[T], q: f64) -> Result<f64, ColumnError> {
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

    let x_i = data[i].as_();
    let x_j = data[j].as_();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_eda() {
        let arr = [1, 2, 3, 4, 5];

        // mean test
        match mean(&arr) {
            Ok(mean) => assert_eq!(mean, 3.),
            Err(_) => panic!("Impossible"),
        };

        match median(&arr) {
            Ok(median) => assert_eq!(median, 3.),
            Err(_) => panic!("Impossible"),
        };
    }

    #[test]
    fn simple_eda_with_floats() {
        // ts exists cus i wanted to see if PartialOrd worked

        let arr = [1., 2., 4., 7., 9.];

        match mean(&arr) {
            Ok(mean) => assert_eq!(mean, 4. + 3. / 5.),
            Err(_) => panic!("Impossible"),
        };

        match median(&arr) {
            Ok(median) => assert_eq!(median, 4.),
            Err(_) => panic!("Impossible"),
        };
    }

    #[test]
    fn even_count_median() {
        let arr = [1, 2, 3, 4, 5, 6];

        match median(&arr) {
            Ok(median) => assert_eq!(median, 3.5),
            Err(_) => panic!("Impossible"),
        };
    }

    #[test]
    fn simple_quantiles() {
        let arr = [1, 2, 3, 4, 5];

        // quartile median test
        match nearest_quantile(&arr, 0.5) {
            Ok(q2) => assert_eq!(q2, 3.),
            Err(_) => panic!("Impossible"),
        };

        // third quartile test
        match nearest_quantile(&arr, 0.75) {
            Ok(q2) => assert_eq!(q2, 4.),
            Err(_) => panic!("Impossible"),
        };
    }

    #[test]
    fn second_quantiles_equals_median() {
        let arr = [6, 7, 8, 9];
        let med = median(&arr).unwrap();
        let q2 = linear_quantile(&arr, 0.5).unwrap();

        assert_eq!(med, q2);
    }

    #[test]
    fn simple_min_and_max() {
        let arr = [1, 2, 3];

        // test a simple minimum
        match min(&arr) {
            Ok(m) => assert_eq!(m, 1),
            Err(_) => panic!("Impossible"),
        }

        // test a simple maximum
        match max(&arr) {
            Ok(m) => assert_eq!(m, 3),
            Err(_) => panic!("Impossible"),
        }
    }
}
