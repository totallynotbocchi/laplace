// Mean Squared Error

pub fn mse(y_pred: &Vec<f64>, y_true: &Vec<f64>) -> f64 {
    if y_pred.len() != y_true.len() {
        panic!("Sizes don't match.")
    }

    let mut sum: f64 = 0.;
    for i in 0..y_pred.len() {
        sum += (y_true[i] - y_pred[i]).powi(2);
    }

    sum / y_pred.len() as f64
}

pub fn mse_grad(y_pred: &Vec<f64>, y_true: &Vec<f64>) -> Vec<f64> {
    if y_pred.len() != y_true.len() {
        panic!("Sizes don't match.")
    }

    let n = y_pred.len() as f64;

    let mut grad = Vec::<f64>::with_capacity(n as usize);
    for i in 0..n as usize {
        grad[i] = 2. / n * (y_pred[i] - y_true[i]);
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    static TOLERANCE: f64 = 0.001;

    #[test]
    fn same_array() {
        let arr1 = vec![0., 1., 2.];

        let err = mse(&arr1, &arr1);
        assert_eq!(err, 0.);
    }

    #[test]
    fn correctness() {
        let y_true = vec![1., 2., 3.];
        let y_pred = vec![1., 2., 5.];

        let err = mse(&y_pred, &y_true);
        assert!((err - 4. / 3.).abs() < TOLERANCE);
    }

    #[test]
    fn negatives() {
        let y_true = vec![-1., -2., -3.];
        let y_pred = vec![-1., -2., -5.];

        let err = mse(&y_pred, &y_true);
        assert!((err - 4. / 3.).abs() < TOLERANCE);
    }
}
