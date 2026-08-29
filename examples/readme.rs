// this is the example used in the root's README.md
use laplace::models::{LinearRegression, Model};

fn main() {
    let x_train = vec![vec![1.], vec![2.], vec![3.], vec![4.]];
    let y_train = vec![2., 4., 6., 8.];

    let mut model = LinearRegression::new(100, 0.1);
    let _ = model.train(&x_train, &y_train);

    let test = vec![3.];
    println!("{:?}", model.predict(&test));
}
