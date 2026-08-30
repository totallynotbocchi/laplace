use crate::stats::EDAError;
use crate::stats::*;
use thiserror::Error;

macro_rules! eda_simple_abst {
    ( $self:ident, $f:ident $(, $args:expr )* ) => {
        match $self {
            Self::Int(v) => $f(&v $(, $args)* ),
            Self::Float(v) => $f(&v $(, $args)* ),
            Self::String(_) => Err(EDAError::WrongType),
        }
    };
}

// macro for dark magic
macro_rules! eda_other_abst {
    ( $self:ident, $other:ident, $f:ident $(, $args:expr)* ) => {
        match ($self, &$other) {
            (Self::Int(v), Self::Int(w)) => Ok($f(v, w $(, $args)*)),
            (Self::Int(v), Self::Float(w)) => Ok($f(v, w $(, $args)*)),
            (Self::Float(v), Self::Int(w)) => Ok($f(v, w $(, $args)*)),
            (Self::Float(v), Self::Float(w)) => Ok($f(v, w $(, $args)*)),
            (_, _) => Err(EDAError::WrongType),
        }
    };
}

// error type
#[derive(Error, Debug, PartialEq, PartialOrd)]
pub enum ColumnError {
    #[error("This column has no data.")]
    Empty,

    #[error("The sizes of the two columns do not match.")]
    NonMatchingSizes,

    #[error("The column is not numeric when it was expected to be,")]
    NonNumerical,
}

pub type ColumnResult<T> = Result<T, ColumnError>;

// a single cell in a column/dataset, used when returning specific column value
#[derive(Clone, Debug)]
pub enum Value {
    Int(i64),
    Float(f64),
    String(String),
}

// a single column of data, only of one type
#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub enum Column {
    Int(Vec<i64>),
    Float(Vec<f64>),
    String(Vec<String>),
}

// TODO: make a macro for creating Columns inline

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
            Self::Float(v) => v.sort_by(|a, b| a.total_cmp(b)), // handle NaN and inf for f64
            Self::String(_) => {}
        };

        self
    }

    pub fn get(&self, idx: usize) -> ColumnResult<Value> {
        match self {
            Column::Int(v) => {
                let value = *v.get(idx).ok_or(ColumnError::NonNumerical)?;
                Ok(Value::Int(value))
            }
            Column::Float(v) => {
                let value = *v.get(idx).ok_or(ColumnError::NonNumerical)?;
                Ok(Value::Float(value))
            }
            Column::String(v) => {
                let value = v.get(idx).ok_or(ColumnError::NonNumerical)?;
                Ok(Value::String(value.clone()))
            }
        }
    }

    // NOTE: this copies data
    pub fn as_f64_vec(&self) -> ColumnResult<Vec<f64>> {
        match self {
            Self::Int(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            Self::Float(v) => Ok(v.clone()),
            Self::String(_) => Err(ColumnError::NonNumerical),
        }
    }

    // ===================== EDA methods =====================

    // =========== min and max ===========

    pub fn max(&self) -> EDAResult<f64> {
        // no macro magic :<
        match self {
            Self::Float(v) => max(v),
            Self::Int(v) => max(v).map(|x| x as f64),
            Self::String(_) => Err(EDAError::WrongType),
        }
    }

    pub fn min(&self) -> EDAResult<f64> {
        match self {
            Self::Float(v) => min(v),
            Self::Int(v) => min(v).map(|x| x as f64),
            Self::String(_) => Err(EDAError::WrongType),
        }
    }

    // =========== central tendency ===========

    pub fn mean(&self) -> EDAResult<f64> {
        // remember, the macro expands to:
        //
        //      match self {
        //          Self::Float(v) => <method>(v, .. ),
        //          Self::Int(v) => <method>(v, ...),
        //          Self::String(_) => Err(EDAError::WrongType),
        //      }

        eda_simple_abst!(self, mean)
    }

    pub fn geometric_mean(&self) -> EDAResult<f64> {
        eda_simple_abst!(self, geometric_mean)
    }

    pub fn trimmed_mean(&self, left: f64, right: f64) -> EDAResult<f64> {
        eda_simple_abst!(self, trimmed_mean, left, right)
    }

    pub fn median(&self) -> EDAResult<f64> {
        eda_simple_abst!(self, median)
    }

    // =========== quantiles and iqr ===========

    pub fn linear_quantile(&self, q: f64) -> EDAResult<f64> {
        eda_simple_abst!(self, linear_quantile, q)
    }

    pub fn nearest_quantile(&self, q: f64) -> EDAResult<f64> {
        eda_simple_abst!(self, nearest_quantile, q)
    }

    pub fn nearest_iqr(&self, q1: f64, q2: f64) -> EDAResult<f64> {
        eda_simple_abst!(self, nearest_iqr, q1, q2)
    }

    pub fn linear_iqr(&self, q1: f64, q2: f64) -> EDAResult<f64> {
        eda_simple_abst!(self, linear_iqr, q1, q2)
    }

    // =========== dispersion ===========

    pub fn pop_var(&self) -> EDAResult<f64> {
        eda_simple_abst!(self, pop_var)
    }

    pub fn samp_var(&self) -> EDAResult<f64> {
        eda_simple_abst!(self, samp_var)
    }

    pub fn pop_std(&self) -> EDAResult<f64> {
        eda_simple_abst!(self, pop_std)
    }

    pub fn samp_std(&self) -> EDAResult<f64> {
        eda_simple_abst!(self, samp_std)
    }

    // =========== correlation and covariance ===========

    pub fn r_corr(&self, other: &Column) -> EDAResult<f64> {
        eda_other_abst!(self, other, r_corr)?
    }

    pub fn pop_cov(&self, other: &Column) -> EDAResult<f64> {
        eda_other_abst!(self, other, pop_cov)?
    }

    pub fn samp_cov(&self, other: &Column) -> EDAResult<f64> {
        eda_other_abst!(self, other, samp_cov)?
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
