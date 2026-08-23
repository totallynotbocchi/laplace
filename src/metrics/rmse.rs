use crate::metrics::mse;

pub fn rmse(y_pred: &[f64], y_true: &[f64]) -> f64 {
    mse(y_pred, y_true).sqrt()
}

pub fn rmse_grad(y_pred: &[f64], y_true: &[f64]) -> Vec<f64> {
    if y_pred.len() != y_true.len() {
        panic!("Sizes don't match.")
    }

    let n = y_pred.len() as f64;

    let mut grad = Vec::<f64>::with_capacity(n as usize);
    for i in 0..n as usize {
        grad[i] = (y_pred[i] - y_true[i]) / (n * (2. / n * (y_pred[i] - y_true[i])));
    }

    grad
}
