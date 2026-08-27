# Laplace

Laplace is a toy ML library written in Rust with (most) features from scratch.

It contains methods and utilities for:

- **Descriptive Statistics**
- **Linear Algebra**
- **Multiple ML Models** (like Linear Regression)
and more to come.

## Usage

For example, using linear regression looks like this:

```rust
use laplace::models::{LinearRegression, Model};

fn main() {
    let x_train = vec![vec![1.], vec![2.], vec![3.], vec![4.]];
    let y_train = vec![2., 4., 6., 8.];

    let mut model = LinearRegression::new(100, 0.1);
    let _ = model.fit(&x_train, &y_train);

    let test = vec![3.];
    println!("{:?}", model.predict(&test));
}
```

## To-Do's

In under no particular order, I am planning:

- [x] Mean and median
- [x] Min, max and quantiles
- [x] Std and var (population and sample)
- [ ] Handling missing data
- [x] Reading CSV data into a `Dataset`
- [x] Linear regression
- [ ] Logistic Regression
- [ ] KNN
- [x] Weighted mean, geometric mean, etc.
- [x] Correlation and covariance
- [ ] Grouping data by column
- [ ] More error/metric functions (like RMSE, R^2 and accuracy)
- [ ] Basic scaling
- [ ] Basic normalizing
- [ ] Outlier handling
- [ ] One-hot encoding
- [ ] Basic probability distributions (sample of a dist., CDF, PDF, etc.)
- [x] Range
- [ ] Tensor and linear algebra methods
