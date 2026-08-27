mod central_tendency;
mod column;
mod correlation;
mod dataset;
mod dispersion;

pub use central_tendency::*;
pub use column::*;
pub use correlation::*;
pub use dataset::*;
pub use dispersion::*;

use thiserror::Error;

// for EDA methods like "mean"
#[derive(Error, Debug)]
pub enum EDAError {
    #[error("The passed array is empty.")]
    EmptyData,

    #[error("The sizes of the passed arrays are different.")]
    DifferentSizes,

    #[error("Bad parameters: {message}")]
    InvalidParameter { message: String },

    #[error("This array has data of the wrong type.")]
    WrongType,
}

pub type EDAResult<T> = Result<T, EDAError>;
