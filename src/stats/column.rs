use std::fmt::Display;

// error type
#[derive(Debug, PartialEq, PartialOrd)]
pub enum ColumnError {
    NonNumerical,
}

// a single column of data, only of one type
#[derive(Debug, PartialEq, PartialOrd)]
pub enum Column {
    Int(Vec<i64>),
    Float(Vec<f64>),
    String(Vec<String>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Self::Int(v) => v.len(),
            Self::Float(v) => v.len(),
            Self::String(v) => v.len(),
        }
    }

    pub fn sort(mut self) -> Self {
        match &mut self {
            Self::Int(v) => v.sort(),
            Self::Float(v) => v.sort_by(|a, b| a.total_cmp(b)),
            Self::String(_) => {}
        };

        self
    }

    // descriptive statistical methods
    pub fn mean(&self) -> Result<f64, ColumnError> {
        let mut sum: f64 = 0.;

        // load the number of elements locally to avoid new function calls and
        // extraneous pattern matching from self.len()
        let n: f64;

        match self {
            Self::Int(v) => {
                v.iter().for_each(|el| sum += *el as f64);
                n = v.len() as f64;
            }
            Self::Float(v) => {
                v.iter().for_each(|el| sum += *el);
                n = v.len() as f64;
            }
            _ => return Err(ColumnError::NonNumerical),
        };

        Ok(sum / n)
    }

    pub fn median(&self) -> Result<f64, ColumnError> {
        let n = self.len();

        match self {
            Self::Int(v) => {
                if n % 2 != 0 {
                    Ok(v[(n - 1) / 2 as usize] as f64)
                } else {
                    Ok((v[n / 2 as usize] + v[n / 2 + 1 as usize]) as f64)
                }
            }
            Self::Float(v) => {
                if n % 2 != 0 {
                    Ok(v[(n - 1) / 2 as usize])
                } else {
                    Ok(v[(n / 2 + n / 2 + 1) as usize])
                }
            }
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    pub fn linear_quantile(&self, q: f64) -> Result<f64, ColumnError> {
        // linear interpolation has the formula:
        //   x_floor(q) + (q - floor(q)) (x_ceil(q) - x_floor(q))
        // where q is the quantile

        let i = q.floor();
        let j = q.ceil();
        let d = q - i;

        match self {
            Self::Int(v) => {
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
            Self::Float(v) => {
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

            Self::String(_) => return Err(ColumnError::NonNumerical),
        }
    }

    pub fn nearest_quantile(&self, q: f64) -> Result<f64, ColumnError> {
        match self {
            Self::Int(v) => {
                let n = v.len();
                let idx = ((n - 1) as f64 * q).round();
                Ok(v[idx as usize] as f64)
            }
            Self::Float(v) => {
                let n = v.len();
                let idx = ((n - 1) as f64 * q).round();
                Ok(v[idx as usize])
            }
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }
}

impl Display for ColumnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg: &str = match self {
            Self::NonNumerical => "This column is non-numerical.",
        };

        write!(f, "Column Error: {}", msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths() {
        let col1 = Column::Int(vec![1, 2, 3]);
        let col2 = Column::Float(vec![1., 2., 3.]);
        let col3 = Column::String(vec!["1".to_string(), "2".to_string(), "3".to_string()]);

        assert_eq!(col1.len(), 3);
        assert_eq!(col2.len(), 3);
        assert_eq!(col3.len(), 3);
    }

    #[test]
    fn sorting() {
        // integers
        let col1 = Column::Int(vec![0, 2, 1, 3]).sort();

        match col1 {
            Column::Int(v) => assert_eq!(v, vec![0, 1, 2, 3]),
            _ => panic!("Impossible"),
        }

        // floats
        let col2 = Column::Float(vec![3.13, 3.14, 3.]).sort();

        match col2 {
            Column::Float(v) => assert_eq!(v, vec![3., 3.13, 3.14]),
            _ => panic!("Impossible"),
        }
    }

    #[test]
    fn simple_eda() {
        let ds = Column::Int(vec![1, 2, 3, 4, 5]);

        // mean test
        match ds.mean() {
            Ok(mean) => assert_eq!(mean, 3.),
            Err(_) => panic!("Impossible"),
        };

        // median test
        match ds.median() {
            Ok(median) => assert_eq!(median, 3.),
            Err(_) => panic!("Impossible"),
        };
    }

    #[test]
    fn simple_quantiles() {
        let ds = Column::Int(vec![1, 2, 3, 4, 5]);

        // quartile median test
        match ds.nearest_quantile(0.5) {
            Ok(q2) => assert_eq!(q2, 3.),
            Err(_) => panic!("Impossible"),
        };

        // third quartile test
        match ds.nearest_quantile(0.75) {
            Ok(q2) => assert_eq!(q2, 4.),
            Err(_) => panic!("Impossible"),
        };
    }
}
