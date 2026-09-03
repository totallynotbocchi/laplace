// Mean Squared Error

pub fn mse(y_pred: &[f64], y_true: &[f64]) -> f64 {
    if y_pred.len() != y_true.len() {
        panic!("Sizes don't match.")
    }

    let mut sum: f64 = 0.;
    for i in 0..y_pred.len() {
        sum += (y_true[i] - y_pred[i]).powi(2);
    }

    sum / y_pred.len() as f64
}

// WARNING: this doesn't divide by n
pub fn mse_grad(y_pred: &[f64], y_true: &[f64]) -> Vec<f64> {
    if y_pred.len() != y_true.len() {
        panic!("Sizes don't match.")
    }

    let grad: Vec<f64> = y_pred
        .iter()
        .zip(y_true)
        .map(|(&pred, &real)| 2. * (pred - real))
        .collect();

    grad
}
