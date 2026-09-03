mod knn_regression;
mod linear_regression;

pub use knn_regression::KNNRegression;
pub use linear_regression::LinearRegression;

use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum ModelError {
    #[error("The input does not match the model's parameters.")]
    SizesNotMatchingParams,

    #[error("There are too little or too many answers to match the input.")]
    TrainAndAnswersSizesDiffer,

    #[error("There is no data.")]
    EmptyData,

    #[error("The weights (not parameters) are not enough to cover every column.")]
    SizesNotMatchingWeights,

    #[error("This model is untrained and cannot predict.")]
    ModelUntrained,
}

type ModelResult<T> = Result<T, ModelError>;

pub trait Model {
    fn train(&mut self, train_data: &Vec<Vec<f64>>, answer_data: &Vec<f64>) -> ModelResult<()>;
    fn predict(&self, data: &Vec<f64>) -> ModelResult<f64>;
}
