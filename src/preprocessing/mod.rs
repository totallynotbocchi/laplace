use crate::stats::{Column, Dataset};
use thiserror::Error;

mod box_cox;
mod function;
mod min_max;
mod polynomial;
mod yeo_johnson;
mod z_score;

pub use box_cox::*;
pub use function::*;
pub use min_max::*;
pub use polynomial::*;
pub use yeo_johnson::*;
pub use z_score::*;

// error type for preprocessors
#[derive(Debug, Error, Clone, Copy)]
pub enum PreprocessingError {
    #[error("There is no data to transform.")]
    EmptyData,

    #[error("The column has the wrong type for this method.")]
    InvalidColumnType,

    #[error("The preprocessor's constansts are uninitialized. Run .fit() first.")]
    UninitializedConstants,

    #[error("Some values in the column break this method (e.g. negative values on Box-Cox).")]
    InvalidColumnValues,

    #[error("The input does not match the size of this transformer's configuration.")]
    MismatchedSizes,

    #[error("The number of columns in the dataset does not match what was expected by the method.")]
    MismatchedColumnCount,
}

pub type PreprocessingResult<T> = Result<T, PreprocessingError>;

// trait type for every column level preprocessor, like min-max
pub trait ColumnPreprocessor {
    // this method makes the struct remember the specific constants of a column so they can be
    // reused for new input
    fn fit(&mut self, data: &Column) -> PreprocessingResult<()>;

    // this method transforms a column with the memorized constants
    fn transform(&self, data: &Column) -> PreprocessingResult<Column>;
}

// trait type for every dataset level preprocessor, like polynomial features
pub trait DatasetPreprocessor {
    fn fit(&mut self, data: &Dataset) -> PreprocessingResult<()>;
    fn transform(&self, data: &Dataset) -> PreprocessingResult<Dataset>;
}
