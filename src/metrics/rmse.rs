// TODO: make this usable bruh

use crate::metrics::mse;

pub fn rmse(y_pred: &[f64], y_true: &[f64]) -> f64 {
    mse(y_pred, y_true).sqrt()
}
