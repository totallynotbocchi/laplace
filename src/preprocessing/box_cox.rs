use crate::{
    preprocessing::{ColumnPreprocessor, PreprocessingError, PreprocessingResult},
    stats::Column,
};

pub struct BoxCoxTransformation {
    lambda: f64,
}

impl BoxCoxTransformation {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    fn transform_value(&self, y: f64) -> f64 {
        if self.lambda == 0. {
            y.ln()
        } else {
            (y.powf(self.lambda) - 1.) / self.lambda
        }
    }
}

impl Default for BoxCoxTransformation {
    fn default() -> Self {
        Self { lambda: 1. }
    }
}

impl ColumnPreprocessor for BoxCoxTransformation {
    // this method only checks if the data has strictly positive numbers
    fn fit(&mut self, data: &Column) -> PreprocessingResult<()> {
        let is_all_positive = data
            .as_f64_vec()
            .map_err(|_| PreprocessingError::InvalidColumnType)?
            .iter()
            .all(|&pt| pt > 0.);

        if is_all_positive {
            return Ok(());
        } else {
            return Err(PreprocessingError::InvalidColumnValues);
        }
    }

    fn transform(&self, data: &Column) -> PreprocessingResult<Column> {
        let v = data
            .as_f64_vec()
            .map_err(|_| PreprocessingError::InvalidColumnType)?
            .iter()
            .map(|&y| self.transform_value(y))
            .collect::<Vec<f64>>();

        Ok(Column::Float(v))
    }
}
