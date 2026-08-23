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
