use crate::metrics::{LossFn, LossGradFn, mse, mse_grad};
use crate::models::{Model, ModelError, ModelResult};

pub struct LinearRegression {
    params: Vec<f64>,
    iters: usize,
    learning_rate: f64,

    loss: LossFn,
    loss_grad: LossGradFn,
}

impl LinearRegression {
    pub fn new(iters: usize, lr: f64) -> Self {
        Self {
            iters,
            loss: mse,           // default
            loss_grad: mse_grad, // default
            params: Vec::new(),
            learning_rate: lr,
        }
    }

    pub fn set_funcs(mut self, loss: LossFn, loss_grad: LossGradFn) -> Self {
        self.loss = loss;
        self.loss_grad = loss_grad;
        self
    }
}

impl Model for LinearRegression {
    fn train(&mut self, train_data: &Vec<Vec<f64>>, answers: &Vec<f64>) -> ModelResult<()> {
        if train_data.len() != answers.len() {
            return Err(ModelError::TrainAndAnswersSizesDiffer);
        } else if train_data.len() == 0 {
            return Err(ModelError::EmptyData);
        }

        // make enough parameters 0
        self.params = vec![0.; train_data[0].len()];

        // run all iterations
        for _ in 1..=self.iters {
            // TODO: make this run in verbose logging mode
            // println!("Iteration no. {iter}");

            // calculate loss per row of input
            let y_pred = train_data // loops thru rows
                .iter()
                .map(|row| self.predict(row).unwrap())
                .collect::<Vec<f64>>();

            // calculate the gradient with the chain rule
            // dL/d(y_pred)
            let loss_grad = mse_grad(&y_pred, answers); // one per sample
            let mut real_grad = vec![0.0; self.params.len()]; // store the final gradient

            // for each row, get derivative and the input data
            for (row, &der) in train_data.iter().zip(&loss_grad) {
                // for each feature in the row and the real, final gradient
                for (real_el, &x) in real_grad.iter_mut().zip(row) {
                    *real_el += der * x; // modify in-place
                }
            }

            // now real_grad contains the gradient with the chain rule (which has the same inner
            // derivative, x_j, for any loss function L)
            // apply gradient descent:
            self.params
                .iter_mut()
                .zip(&real_grad)
                .for_each(|(param, dl)| *param -= self.learning_rate * dl);
        }

        Ok(())
    }

    fn predict(&self, data: &Vec<f64>) -> ModelResult<f64> {
        if data.len() != self.params.len() {
            return Err(ModelError::SizesNotMatchingParams);
        }

        Ok(data.iter().zip(&self.params).map(|(&x, &w)| x * w).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overfitting() {
        let x = vec![vec![1.], vec![2.], vec![3.], vec![4.]]; // 4 rows, 1 feature each
        let y = vec![2., 4., 6., 8.]; // 4 answers

        let mut model = LinearRegression::new(500, 0.01);
        let _ = model.train(&x, &y);

        let test = vec![3.];
        println!("{:?}", model.predict(&test));
    }
}
