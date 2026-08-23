mod mse;
mod rmse;

pub use mse::mse;
pub use mse::mse_grad;
pub use rmse::rmse;

type LossFn = fn(&[f64], &[f64]) -> f64;
type LossGradFn = fn(&[f64], &[f64]) -> Vec<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    static TOLERANCE: f64 = 0.0001;
    static ERR_FUNCS: [LossFn; 1] = [mse];

    #[test]
    fn same_array() {
        let arr1 = [0., 1., 2.];

        let correct_errors = [0.];
        for i in 0..ERR_FUNCS.len() {
            assert_eq!(ERR_FUNCS[i](&arr1, &arr1), correct_errors[i]);
        }
    }

    #[test]
    fn correctness() {
        let y_true = [1., 2., 3.];
        let y_pred = [1., 2., 5.];

        let correct_errors = [4. / 3.];
        for i in 0..ERR_FUNCS.len() {
            let err = ERR_FUNCS[i](&y_pred, &y_true);
            assert!((err - correct_errors[i]).abs() < TOLERANCE);
        }
    }

    #[test]
    fn negatives() {
        let y_true = [-1., -2., -3.];
        let y_pred = [-1., -2., -5.];

        let correct_errors = [4. / 3.];
        for i in 0..ERR_FUNCS.len() {
            let err = ERR_FUNCS[i](&y_pred, &y_true);
            assert!((err - correct_errors[i]).abs() < TOLERANCE);
        }
    }
}
