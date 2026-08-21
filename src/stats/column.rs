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
}
