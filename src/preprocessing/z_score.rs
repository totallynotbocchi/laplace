use crate::{
    preprocessing::{PreprocessingError, PreprocessingResult, Preprocessor},
    stats::Column,
};

// scaling by using a point's z-score
pub struct ZScoreScaler {
    std: Option<f64>,
    mean: Option<f64>,
}

impl ZScoreScaler {
    pub fn new() -> Self {
        Self {
            std: None,
            mean: None,
        }
    }
}

impl Preprocessor for ZScoreScaler {
    fn fit(&mut self, data: &Column) -> PreprocessingResult<()> {
        self.mean = Some(
            data.mean()
                .map_err(|_| PreprocessingError::InvalidColumnType)?,
        );

        self.std = Some(
            data.pop_std()
                .map_err(|_| PreprocessingError::InvalidColumnType)?,
        );

        Ok(())
    }

    fn transform(&self, data: &Column) -> PreprocessingResult<Column> {
        let mean = self
            .mean
            .ok_or(PreprocessingError::UninitializedConstants)?;
        let std = self.std.ok_or(PreprocessingError::UninitializedConstants)?;

        let arr = data
            .as_f64_vec()
            .map_err(|_| PreprocessingError::InvalidColumnType)?
            .iter()
            .map(|x| (x - mean) / std)
            .collect::<Vec<f64>>();

        Ok(Column::Float(arr))
    }
}
