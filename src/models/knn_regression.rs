use crate::models::{Model, ModelResult};

#[derive(Debug, Clone, Copy)]
pub enum DistanceMethod {
    Euclidean,
}

// model struct
pub struct KNNRegression {
    k: usize,
    distance_method: DistanceMethod,

    train_data: Vec<Vec<f64>>,
    answers: Vec<f64>,
}

impl KNNRegression {
    pub fn new(k: usize, distance_method: DistanceMethod) -> Self {
        Self {
            k,
            distance_method,

            train_data: Vec::new(),
            answers: Vec::new(),
        }
    }

    fn euclidean(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn get_distance_method(&self) -> fn(&[f64], &[f64]) -> f64 {
        match self.distance_method {
            DistanceMethod::Euclidean => Self::euclidean,
        }
    }
}

impl Model for KNNRegression {
    // NOTE: this copies data
    fn fit(&mut self, train_data: &Vec<Vec<f64>>, answer_data: &Vec<f64>) -> ModelResult<()> {
        self.train_data = train_data.clone();
        self.answers = answer_data.clone();
        Ok(())
    }

    fn predict(&self, data: &Vec<f64>) -> ModelResult<f64> {
        // calculate distances from each point in data
        // store index and distance
        let mut distances = self
            .train_data
            .iter()
            .enumerate()
            .map(|(i, tr)| (i, (self.get_distance_method())(tr, data)))
            .collect::<Vec<(usize, f64)>>();

        // sort descending by distance (.1 in the tuple)
        distances.sort_by(|a, b| (a.1).total_cmp(&b.1));

        // get the k smallest distances and average the labels they point to
        let sum = &distances[0..self.k]
            .iter()
            .map(|(i, _)| self.answers[*i])
            .sum::<f64>();

        Ok(sum / self.k as f64)
    }
}

// TODO: tests
