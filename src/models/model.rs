use crate::stats::{column::Column, dataset::Dataset};

pub trait Model {
    fn fit(&mut self, train_data: &Dataset, answer_data: &Column);
    fn predict(&self, data: &Dataset) -> i64;
}
