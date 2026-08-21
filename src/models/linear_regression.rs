use crate::{
    models::model::Model,
    stats::{column::Column, dataset::Dataset},
};

pub struct LinearRegression {
    params: Vec<i64>,
    iters: usize,
}

impl LinearRegression {
    pub fn new(iters: usize) -> Self {
        let params = Vec::<i64>::new();

        Self { iters, params }
    }
}

impl Model for LinearRegression {
    fn fit(&mut self, train_data: &Dataset, answer_data: &Column) {
        // TODO: yes
    }

    fn predict(&self, data: &Dataset) -> i64 {
        // TODO: yes
        0
    }
}
