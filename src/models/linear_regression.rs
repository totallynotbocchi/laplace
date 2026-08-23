// TODO: implement
// this should be the first model that has a Config with hyperoperations

use crate::{
    models::Model,
    stats::{Column, Dataset},
};

pub struct LinearRegression {
    _params: Vec<i64>,
    _iters: usize,
}

impl LinearRegression {
    pub fn new(_iters: usize) -> Self {
        let _params = Vec::<i64>::new();

        Self { _iters, _params }
    }
}

impl Model for LinearRegression {
    fn fit(&mut self, _train_data: &Dataset, _answer_data: &Column) {
        // TODO: yes
    }

    fn predict(&self, _data: &Dataset) -> i64 {
        // TODO: yes
        0
    }
}
