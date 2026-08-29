use crate::{
    preprocessing::{PreprocessingError, PreprocessingResult, Preprocessor},
    stats::Column,
};

pub struct YeoJohnsonTransformation {
    lambda: f64,
}

impl YeoJohnsonTransformation {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    pub fn default() -> Self {
        Self { lambda: 1. }
    }
}

impl Preprocessor for YeoJohnsonTransformation {
    // because yeo-johnson works for every value y, this method only checks if the column type is
    // valid, we do this here as well so type errors are caught earlier
    fn fit(&mut self, data: &Column) -> PreprocessingResult<()> {
        data.as_f64_vec()
            .map_err(|_| PreprocessingError::InvalidColumnType)?;

        Ok(())
    }

    fn transform(&self, data: &Column) -> PreprocessingResult<Column> {
        let v = data
            .as_f64_vec()
            .map_err(|_| PreprocessingError::InvalidColumnType)?
            .iter()
            .map(|&y| {
                if y >= 0. && self.lambda == 0. {
                    (y + 1.).ln()
                } else if y >= 0. {
                    ((y + 1.).powf(self.lambda) - 1.) / self.lambda
                } else if y < 0. && self.lambda == 2. {
                    -(-y + 1.).ln()
                } else {
                    -((-y + 1.).powf(2. - self.lambda) - 1.) / (2. - self.lambda)
                }
            })
            .collect::<Vec<f64>>();

        Ok(Column::Float(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static TOLERANCE: f64 = 0.0001;

    #[test]
    fn simple() {
        let col = Column::Float(vec![1., 2., 3., 4.]);
        let actual = vec![0.82842712, 1.46410162, 2., 2.47213595];

        let mut yj = YeoJohnsonTransformation::new(0.5);
        yj.fit(&col).unwrap();

        let new_col = yj.transform(&col).unwrap();
        match new_col {
            Column::Float(output) => output
                .iter()
                .zip(actual)
                .map(|(&this, real)| assert!((real - this).abs() < TOLERANCE))
                .collect::<Vec<()>>(), // make the iterator run
            _ => panic!("Impossible"),
        };
    }

    #[test]
    fn negatives() {
        let col = Column::Float(vec![-1., -2., 3., 4.]);
        let actual = vec![-1.21895142, -2.79743495, 2., 2.47213595];

        let mut yj = YeoJohnsonTransformation::new(0.5);
        yj.fit(&col).unwrap();

        let new_col = yj.transform(&col).unwrap();
        match new_col {
            Column::Float(output) => output
                .iter()
                .zip(actual)
                .map(|(&neg, real)| assert!((neg - real).abs() < TOLERANCE))
                .collect::<Vec<()>>(), // make the iterator run
            _ => panic!("Impossible"),
        };
    }
}
