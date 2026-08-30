use crate::{
    preprocessing::{ColumnPreprocessor, PreprocessingError, PreprocessingResult},
    stats::Column,
};

pub struct MinMaxScaler {
    min: Option<f64>,
    max: Option<f64>,
}

impl MinMaxScaler {
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
        }
    }
}

impl ColumnPreprocessor for MinMaxScaler {
    fn fit(&mut self, data: &Column) -> PreprocessingResult<()> {
        if data.len() == 0 {
            return Err(PreprocessingError::EmptyData);
        }

        self.min = Some(
            data.min()
                .map_err(|_| PreprocessingError::InvalidColumnType)?,
        );

        self.max = Some(
            data.max()
                .map_err(|_| PreprocessingError::InvalidColumnType)?,
        );

        Ok(())
    }

    fn transform(&self, data: &Column) -> PreprocessingResult<Column> {
        if data.len() == 0 {
            return Err(PreprocessingError::EmptyData);
        }

        let min = self.min.ok_or(PreprocessingError::UninitializedConstants)?;
        let max = self.max.ok_or(PreprocessingError::UninitializedConstants)?;

        let arr = data
            .as_f64_vec()
            .map_err(|_| PreprocessingError::InvalidColumnType)?
            .iter()
            .map(|x| (x - min) / (max - min))
            .collect::<Vec<f64>>();

        Ok(Column::Float(arr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let col = Column::Float(vec![1., 2., 3., 4.]);

        let mut scaler = MinMaxScaler::new();
        scaler.fit(&col).unwrap();

        let new_col = scaler.transform(&col).unwrap();
        assert_eq!(
            new_col.as_f64_vec().unwrap(),
            vec![0., 1. / 3., 2. / 3., 1.]
        );
    }
}
