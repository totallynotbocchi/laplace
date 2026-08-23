pub mod linear_regression;

pub use linear_regression::LinearRegression;

use crate::stats::{Column, Dataset};

pub trait Model {
    fn fit(&mut self, train_data: &Dataset, answer_data: &Column);
    fn predict(&self, data: &Dataset) -> i64;
}
