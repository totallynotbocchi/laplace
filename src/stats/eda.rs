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

    match col {
        Column::Int(v) => {
            if n % 2 != 0 {
                Ok(v[(n - 1) / 2 as usize] as f64)
            } else {
                Ok((v[n / 2 as usize] + v[n / 2 + 1 as usize]) as f64)
            }
        }
        Column::Float(v) => {
            if n % 2 != 0 {
                Ok(v[(n - 1) / 2 as usize])
            } else {
                Ok(v[(n / 2 + n / 2 + 1) as usize])
            }
        }
        Column::String(_) => Err(ColumnError::NonNumerical),
    }
}

pub fn linear_quantile(col: &Column, q: f64) -> Result<f64, ColumnError> {
    // linear interpolation has the formula:
    //   x_floor(q) + (q - floor(q)) (x_ceil(q) - x_floor(q))
    // where q is the quantile

    let i = q.floor();
    let j = q.ceil();
    let d = q - i;

    match col {
        Column::Int(v) => {
            let x_i = v[i as usize] as f64;
            let x_j = v[j as usize] as f64;

            // handle simple edge cases
            if q == 0. {
                return Ok(v[0] as f64);
            } else if q == 0.5 {
                return Ok(x_i + x_j / 2.);
            }

            let x_diff = x_j - x_i;
            Ok(x_i + (d * x_diff as f64))
        }
        Column::Float(v) => {
            let x_i = v[i as usize] as f64;
            let x_j = v[j as usize] as f64;

            // handle simple edge cases
            if q == 0. {
                return Ok(v[0] as f64);
            } else if q == 0.5 {
                return Ok(x_i + x_j / 2.);
            }

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
        let ds = Column::Int(vec![1, 2, 3, 4, 5]);

        // mean test
        match mean(&ds) {
            Ok(mean) => assert_eq!(mean, 3.),
            Err(_) => panic!("Impossible"),
        };

        // median test
        match median(&ds) {
            Ok(median) => assert_eq!(median, 3.),
            Err(_) => panic!("Impossible"),
        };
    }

    #[test]
    fn simple_quantiles() {
        let ds = Column::Int(vec![1, 2, 3, 4, 5]);

        // quartile median test
        match nearest_quantile(&ds, 0.5) {
            Ok(q2) => assert_eq!(q2, 3.),
            Err(_) => panic!("Impossible"),
        };

        // third quartile test
        match nearest_quantile(&ds, 0.75) {
            Ok(q2) => assert_eq!(q2, 4.),
            Err(_) => panic!("Impossible"),
        };
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
