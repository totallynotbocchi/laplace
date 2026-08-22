use crate::stats::column::{Column, ColumnError};

pub fn min(col: &Column) -> Result<f64, ColumnError> {
    let mut min: f64 = f64::MAX;

    let data: &[f64];
    let float_v: Vec<f64>;
    match col {
        Column::Int(v) => {
            float_v = v.iter().map(|x| *x as f64).collect();
            data = &float_v;
        }
        Column::Float(v) => data = v.as_slice(),
        Column::String(_) => return Err(ColumnError::NonNumerical),
    };

    // linear scan for minimum
    for x in data {
        if *x <= min {
            min = *x
        }
    }

    Ok(min)
}

pub fn max(col: &Column) -> Result<f64, ColumnError> {
    let mut max: f64 = f64::MIN;

    let data: &[f64];
    let float_v: Vec<f64>;
    match col {
        Column::Int(v) => {
            float_v = v.iter().map(|x| *x as f64).collect();
            data = &float_v;
        }
        Column::Float(v) => data = v.as_slice(),
        Column::String(_) => return Err(ColumnError::NonNumerical),
    };

    // linear scan for maximum
    for x in data {
        if *x >= max {
            max = *x
        }
    }

    Ok(max)
}

pub fn mean(col: &Column) -> Result<f64, ColumnError> {
    let mut sum: f64 = 0.;

    // load the number of elements locally to avoid new function calls and
    // extraneous pattern matching from self.len()
    let n = col.len() as f64;

    match col {
        Column::Int(v) => v.iter().for_each(|el| sum += *el as f64),
        Column::Float(v) => v.iter().for_each(|el| sum += *el),
        _ => return Err(ColumnError::NonNumerical),
    };

    Ok(sum / n)
}

pub fn median(col: &Column) -> Result<f64, ColumnError> {
    let n = col.len();

    let med: f64;
    match col {
        Column::Int(v) => {
            if n % 2 == 0 {
                med = (v[n / 2 - 1] as f64 + v[n / 2] as f64) / 2.;
            } else {
                med = v[n / 2] as f64;
            }

            Ok(med)
        }
        Column::Float(v) => {
            if n % 2 == 0 {
                med = (v[n / 2 - 1] + v[n / 2]) / 2.;
            } else {
                med = v[n / 2];
            }

            Ok(med)
        }
        Column::String(_) => Err(ColumnError::NonNumerical),
    }
}

pub fn linear_quantile(col: &Column, q: f64) -> Result<f64, ColumnError> {
    // linear interpolation has the formula:
    //   x_floor(q) + (q - floor(q)) (x_ceil(q) - x_floor(q))
    // where q is the quantile

    let h: f64 = q * ((col.len() - 1) as f64);

    let i: usize = h.floor() as usize;
    let j: usize = h.ceil() as usize;
    let d: f64 = h - i as f64;

    match col {
        Column::Int(v) => {
            let x_i = v[i] as f64;
            let x_j = v[j] as f64;

            let x_diff = x_j - x_i;
            Ok(x_i + (d * x_diff as f64))
        }
        Column::Float(v) => {
            let x_i = v[i] as f64;
            let x_j = v[j] as f64;

            let x_diff = x_j - x_i;
            Ok(x_i + (d * x_diff as f64))
        }

        Column::String(_) => return Err(ColumnError::NonNumerical),
    }
}

pub fn nearest_quantile(col: &Column, q: f64) -> Result<f64, ColumnError> {
    match col {
        Column::Int(v) => {
            let n = v.len();
            let idx = ((n - 1) as f64 * q).round();
            Ok(v[idx as usize] as f64)
        }
        Column::Float(v) => {
            let n = v.len();
            let idx = ((n - 1) as f64 * q).round();
            Ok(v[idx as usize])
        }
        Column::String(_) => Err(ColumnError::NonNumerical),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_eda() {
        let col = Column::Int(vec![1, 2, 3, 4, 5]);

        // mean test
        match mean(&col) {
            Ok(mean) => assert_eq!(mean, 3.),
            Err(_) => panic!("Impossible"),
        };

        // median test
        match median(&col) {
            Ok(median) => assert_eq!(median, 3.),
            Err(_) => panic!("Impossible"),
        };
    }

    #[test]
    fn even_count_median() {
        let col = Column::Int(vec![1, 2, 3, 4, 5, 6]);

        match median(&col) {
            Ok(median) => assert_eq!(median, 3.5),
            Err(_) => panic!("Impossible"),
        };
    }

    #[test]
    fn simple_quantiles() {
        let col = Column::Int(vec![1, 2, 3, 4, 5]);

        // quartile median test
        match nearest_quantile(&col, 0.5) {
            Ok(q2) => assert_eq!(q2, 3.),
            Err(_) => panic!("Impossible"),
        };

        // third quartile test
        match nearest_quantile(&col, 0.75) {
            Ok(q2) => assert_eq!(q2, 4.),
            Err(_) => panic!("Impossible"),
        };
    }

    #[test]
    fn second_quantiles_equals_median() {
        let col = Column::Int(vec![6, 7, 8, 9]);
        let med = median(&col).unwrap();
        let q2 = linear_quantile(&col, 0.5).unwrap();

        assert_eq!(med, q2);
    }

    #[test]
    fn simple_min_and_max() {
        let col = Column::Int(vec![1, 2, 3]);

        // test a simple minimum
        match min(&col) {
            Ok(m) => assert_eq!(m, 1.),
            Err(_) => panic!("Impossible"),
        }

        // test a simple maximum
        match max(&col) {
            Ok(m) => assert_eq!(m, 3.),
            Err(_) => panic!("Impossible"),
        }
    }
}
