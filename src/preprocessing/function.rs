use crate::{
    preprocessing::{PreprocessingError, PreprocessingResult, Preprocessor},
    stats::Column,
};

pub struct FunctionTransformation {
    func: fn(f64) -> f64,
    inverse: fn(f64) -> f64,
}

impl FunctionTransformation {
    pub fn new(func: fn(f64) -> f64, inverse: fn(f64) -> f64) -> Self {
        Self { func, inverse }
    }

    fn inverse_transform(&self, data: &Column) -> PreprocessingResult<Column> {
        let v = data
            .as_f64_vec()
            .map_err(|_| PreprocessingError::InvalidColumnType)?
            .iter()
            .map(|&x| (self.inverse)(x))
            .collect::<Vec<f64>>();

        Ok(Column::Float(v))
    }
}

impl Preprocessor for FunctionTransformation {
    // does nothing
    fn fit(&mut self, _data: &Column) -> PreprocessingResult<()> {
        Ok(())
    }

    fn transform(&self, data: &Column) -> PreprocessingResult<Column> {
        let v = data
            .as_f64_vec()
            .map_err(|_| PreprocessingError::InvalidColumnType)?
            .iter()
            .map(|&x| (self.func)(x))
            .collect::<Vec<f64>>();

        Ok(Column::Float(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static TOLERANCE: f64 = 0.00001;

    #[test]
    fn natural_log() {
        let col = Column::Int(vec![1, 2, 3, 4, 5]);

        let ft = FunctionTransformation::new(f64::ln, f64::exp);
        assert_eq!(
            ft.transform(&col).unwrap().as_f64_vec().unwrap(),
            vec![0., 2_f64.ln(), 3_f64.ln(), 4_f64.ln(), 5_f64.ln()]
        );
    }

    #[test]
    fn exponential_inverse() {
        let col = Column::Float(vec![0., 2_f64.ln(), 3_f64.ln(), 4_f64.ln(), 5_f64.ln()]);
        let actual = vec![1., 2., 3., 4., 5.];

        let ft = FunctionTransformation::new(f64::ln, f64::exp);

        // NOTE: we use tolerance here because of floating point errors
        let new_vec = ft.inverse_transform(&col).unwrap().as_f64_vec().unwrap();
        let _ = new_vec
            .iter()
            .zip(actual)
            .map(|(&exp_result, real)| assert!((exp_result - real).abs() < TOLERANCE))
            .collect::<Vec<()>>(); // make the iterator run
    }
}
