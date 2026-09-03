use crate::metrics::{LossFn, LossGradFn, mse, mse_grad};
use crate::models::{Model, ModelError, ModelResult};

pub struct LinearRegression {
    params: Vec<f64>,
    weights: Vec<f64>,
    iters: usize,
    learning_rate: f64,

    loss: LossFn,
    loss_grad: LossGradFn,
}

impl LinearRegression {
    pub fn new(iters: usize, lr: f64) -> Self {
        Self {
            iters,
            loss: mse,           // default function
            loss_grad: mse_grad, // default function
            params: Vec::default(),
            weights: Vec::default(),
            learning_rate: lr,
        }
    }

    pub fn set_funcs(mut self, loss: LossFn, loss_grad: LossGradFn) -> Self {
        self.loss = loss;
        self.loss_grad = loss_grad;
        self
    }

    pub fn set_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = weights;
        self
    }
}

impl Model for LinearRegression {
    // train data is row-major (list of rows)
    // rows = examples
    fn train(&mut self, train_data: &Vec<Vec<f64>>, answers: &Vec<f64>) -> ModelResult<()> {
        if train_data.len() != answers.len() {
            return Err(ModelError::TrainAndAnswersSizesDiffer);
        } else if train_data.len() == 0 {
            return Err(ModelError::EmptyData);
        }

        // make enough parameters 0
        self.params = vec![0.; train_data[0].len() + 1]; // +1 for the bias term

        // if there are no weights, set them to 1
        if self.weights.is_empty() {
            self.weights = vec![1.; train_data.len()];
        } else if self.weights.len() != train_data.len() {
            return Err(ModelError::SizesNotMatchingWeights);
        }

        // run all iterations
        for _ in 1..=self.iters {
            // TODO: make this run in verbose logging mode
            // println!("Iteration no. {iter}");

            // make the predictions per rows
            // index i => prediction for row i
            let y_preds: Vec<f64> = train_data
                .iter()
                .map(|row| self.predict(row))
                .collect::<Result<Vec<f64>, ModelError>>()?;

            // compute the loss gradient per row
            let loss = (self.loss_grad)(&y_preds, answers);

            // apply the chain rule and multiply by dJ(x^i)/dw_j = x^i_j during the sum
            let mut grad: Vec<f64> = vec![0.; self.params.len()]; // gradient has #features size

            // precompute the sum of weights
            let weights_sum: f64 = self.weights.iter().sum();

            // for each parameter
            for j in 0..self.params.len() {
                // for each training example
                for i in 0..train_data.len() {
                    // account for the bias term
                    if j == self.params.len() - 1 {
                        grad[j] += loss[i] * (self.weights[i] / weights_sum);
                    } else {
                        grad[j] += loss[i] * train_data[i][j] * (self.weights[i] / weights_sum);
                    }
                }
            }

            // apply gradient descent
            self.params
                .iter_mut()
                .enumerate()
                .for_each(|(i, param)| *param -= self.learning_rate * grad[i]);
        }

        Ok(())
    }

    fn predict(&self, data: &Vec<f64>) -> ModelResult<f64> {
        if self.params.len() == 0 {
            return Err(ModelError::ModelUntrained);
        } else if data.len() != self.params.len() - 1 {
            return Err(ModelError::SizesNotMatchingParams);
        }

        let sum = self
            .params
            .iter()
            .zip(data)
            .map(|(theta, &x)| theta * x)
            .sum::<f64>();
        let bias = self.params[self.params.len() - 1];

        Ok(sum + bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static TOLERANCE: f64 = 0.001;

    // NOTE: these tests were fact checked by scikit-learn but i didint include them in this
    // repository, i delete em

    #[test]
    fn overfitting() {
        let x = vec![vec![1.], vec![2.], vec![3.], vec![4.]]; // 4 rows, 1 feature each
        let y = vec![2., 4., 6., 8.]; // 4 answers, 2x

        let mut model = LinearRegression::new(500, 0.1);
        model.train(&x, &y).unwrap();

        let test = vec![3.];
        let pred = model.predict(&test).unwrap();
        println!("{}", pred);

        assert!((6. - pred).abs() < TOLERANCE);
    }

    #[test]
    fn bias_term() {
        let x = vec![vec![1.], vec![2.], vec![3.], vec![4.]]; // 4 rows, 1 feature each
        let y = vec![3., 5., 7., 9.]; // 4 answers, 2x + 1

        let mut model = LinearRegression::new(500, 0.1);
        model.train(&x, &y).unwrap();

        let test = vec![5.];
        let pred = model.predict(&test).unwrap();
        println!("{}", pred);

        assert!((11. - pred).abs() < TOLERANCE);
    }

    #[test]
    fn weights() {
        let x = vec![vec![1.], vec![2.], vec![3.], vec![4.]];
        let y = vec![2., 4., 6., 8.];
        let weights = vec![6., 7., 9., 9.];

        let mut model = LinearRegression::new(500, 0.1).set_weights(weights);
        model.train(&x, &y).unwrap();

        let test = vec![3.];
        let pred = model.predict(&test).unwrap();
        println!("{}", pred);

        assert!((6. - pred).abs() < TOLERANCE);
    }
}
