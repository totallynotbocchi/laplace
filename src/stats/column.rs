use std::fmt::Display;

use crate::stats::*;

// error type
#[derive(Debug, PartialEq, PartialOrd)]
pub enum ColumnError {
    NonNumerical,
    Empty,
    NonMatchingSizes,
    InvalidInput,
}

impl Display for ColumnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg: &str = match self {
            Self::NonNumerical => "This column has non-numerical data.",
            Self::Empty => "This column has no data.",
            Self::NonMatchingSizes => "The sizes of the two columns do not match.",
            Self::InvalidInput => "The input given to this function is invalid given this data.",
        };

        write!(f, "Column Error: {}", msg)
    }
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

    // ===================== EDA methods =====================

    // =========== min and max ===========

    pub fn max(&self) -> Result<f64, ColumnError> {
        match self {
            Self::Float(v) => max(v),
            Self::Int(v) => max(v).map(|x| x as f64),
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    pub fn min(&self) -> Result<f64, ColumnError> {
        match self {
            Self::Float(v) => min(v),
            Self::Int(v) => min(v).map(|x| x as f64),
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    // =========== central tendency ===========

    pub fn mean(&self) -> Result<f64, ColumnError> {
        match self {
            Self::Float(v) => mean(v),
            Self::Int(v) => mean(v),
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    pub fn median(&self) -> Result<f64, ColumnError> {
        match self {
            Self::Float(v) => median(v),
            Self::Int(v) => median(v),
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    // =========== quantiles and iqr ===========

    pub fn linear_quantile(&self, q: f64) -> Result<f64, ColumnError> {
        match self {
            Self::Float(v) => linear_quantile(v, q),
            Self::Int(v) => linear_quantile(v, q),
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    pub fn nearest_quantile(&self, q: f64) -> Result<f64, ColumnError> {
        match self {
            Self::Float(v) => nearest_quantile(v, q),
            Self::Int(v) => nearest_quantile(v, q),
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    pub fn nearest_iqr(&self, q1: f64, q2: f64) -> Result<f64, ColumnError> {
        match self {
            Self::Float(v) => nearest_iqr(v, q1, q2),
            Self::Int(v) => nearest_iqr(v, q1, q2),
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    pub fn linear_iqr(&self, q1: f64, q2: f64) -> Result<f64, ColumnError> {
        match self {
            Self::Float(v) => linear_iqr(v, q1, q2),
            Self::Int(v) => linear_iqr(v, q1, q2),
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    // =========== dispersion ===========

    pub fn pop_var(&self) -> Result<f64, ColumnError> {
        match self {
            Self::Float(v) => pop_var(v),
            Self::Int(v) => pop_var(v),
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    pub fn samp_var(&self) -> Result<f64, ColumnError> {
        match self {
            Self::Float(v) => samp_var(v),
            Self::Int(v) => samp_var(v),
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    pub fn pop_std(&self) -> Result<f64, ColumnError> {
        match self {
            Self::Float(v) => pop_std(v),
            Self::Int(v) => pop_std(v),
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    pub fn samp_std(&self) -> Result<f64, ColumnError> {
        match self {
            Self::Float(v) => samp_std(v),
            Self::Int(v) => samp_std(v),
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    // =========== correlation and covariance ===========

    // i wish i didint have to copy paste but it would be template spam
    pub fn r_corr(&self, other: &Column) -> Result<f64, ColumnError> {
        // dark magic
        match (self, &other) {
            (Self::Int(v), Self::Int(w)) => Ok(r_corr(v, w)),
            (Self::Int(v), Self::Float(w)) => Ok(r_corr(v, w)),
            (Self::Float(v), Self::Int(w)) => Ok(r_corr(v, w)),
            (Self::Float(v), Self::Float(w)) => Ok(r_corr(v, w)),
            (_, _) => Err(ColumnError::NonNumerical),
        }?
    }

    pub fn pop_cov(&self, other: &Column) -> Result<f64, ColumnError> {
        match (self, &other) {
            (Self::Int(v), Self::Int(w)) => Ok(pop_cov(v, w)),
            (Self::Int(v), Self::Float(w)) => Ok(pop_cov(v, w)),
            (Self::Float(v), Self::Int(w)) => Ok(pop_cov(v, w)),
            (Self::Float(v), Self::Float(w)) => Ok(pop_cov(v, w)),
            (_, _) => Err(ColumnError::NonNumerical),
        }?
    }

    pub fn samp_cov(&self, other: &Column) -> Result<f64, ColumnError> {
        match (self, &other) {
            (Self::Int(v), Self::Int(w)) => Ok(samp_cov(v, w)),
            (Self::Int(v), Self::Float(w)) => Ok(samp_cov(v, w)),
            (Self::Float(v), Self::Int(w)) => Ok(samp_cov(v, w)),
            (Self::Float(v), Self::Float(w)) => Ok(samp_cov(v, w)),
            (_, _) => Err(ColumnError::NonNumerical),
        }?
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
}
