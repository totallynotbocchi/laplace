use crate::models::{Model, ModelError, ModelResult};

#[derive(Debug, Clone, Copy)]
pub enum DistanceMethod {
    Euclidean,
}

// model struct
pub struct KNNRegression {
    k: usize,
    distance_method: DistanceMethod,

    train_data: Vec<Vec<f64>>,
    answer_data: Vec<f64>,
}

impl KNNRegression {
    pub fn new(k: usize, distance_method: DistanceMethod) -> Self {
        Self {
            k,
            distance_method,

            train_data: Vec::new(),
            answer_data: Vec::new(),
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
        if train_data.len() != answer_data.len() {
            return Err(ModelError::TrainAndAnswersSizesDiffer);
        } else if train_data.len() == 0 {
            return Err(ModelError::EmptyData);
        }

        self.train_data = train_data.clone();
        self.answer_data = answer_data.clone();
        Ok(())
    }

    fn predict(&self, data: &Vec<f64>) -> ModelResult<f64> {
        if data.len() != self.train_data[0].len() {
            return Err(ModelError::SizesNotMatchingParams);
        } else if data.len() == 0 {
            return Err(ModelError::EmptyData);
        }

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
            .map(|(i, _)| self.answer_data[*i])
            .sum::<f64>();

        Ok(sum / self.k as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_tie() {
        // one input x, one output
        let x = vec![vec![1.], vec![2.], vec![6.], vec![10.]];
        let y = vec![1., 3., 2., 5.];

        let mut knn = KNNRegression::new(2, DistanceMethod::Euclidean);
        knn.fit(&x, &y).unwrap();

        let pred = knn.predict(&vec![4.5]).unwrap();
        assert_eq!(pred, 2.5);
    }
}
