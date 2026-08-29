<p align="center">
  <img src="./media/icon.svg" width="26%">
</p>

# Laplace

Laplace is a toy ML/math library written in Rust, mostly from scratch.

Right now it contains utilities for:

- **Descriptive statistics**
- **Linear algebra**
- **Machine learning**

## About

I made this library in order to learn and apply AI/ML, data science and math concepts in the most fun way I could think of.

I do not expect this to be particularly useful in production later, I only see this as a personal, small project for educational purposes.

## Usage

For example, using linear regression looks like this:

```rust
use laplace::models::{LinearRegression, Model};

fn main() {
    let x_train = vec![vec![1.], vec![2.], vec![3.], vec![4.]];
    let y_train = vec![2., 4., 6., 8.];

    let mut model = LinearRegression::new(100, 0.1);
    let _ = model.train(&x_train, &y_train).unwrap();

    let test = vec![3.];
    println!("{:?}", model.predict(&test));
}
```

The output to this code is:

```txt
Ok(6.0)
```

## Examples

There are a few examples you can run and read to understand how to use the library in the `/examples` folder.

To run the example in the _Usage_ section for yourself, run:

```txt
cargo run --example readme
```

## Roadmap

The features that I want to try to add are roughly:

- Descriptive statistics
  - [x] Mean and median
  - [x] Min, max and quantiles
  - [x] Std and var (population and sample)
  - [x] Range
  - [x] Weighted mean, geometric mean, etc.
  - [x] Correlation and covariance
- Preprocessing
  - [x] Reading CSV data into a `Dataset`
  - [ ] Grouping data by column
  - [ ] Handling missing data
  - [x] Feature scaling
  - [x] Feature transformation
  - [ ] Outlier handling
  - [ ] One-hot encoding
  - [ ] Generalized function transformations
- Models
  - [x] Linear regression
  - [x] KNN
  - [ ] More error/metric functions (like RMSE, R^2 and accuracy)
  - [ ] Classification variants (for KNN, linear reg., etc.)
- Probability
  - [ ] Basic probability distributions (with samples of dists. and etc.)
  - [ ] Probability distribution methods (like CDF, PDF, etc.)
- Combinatorics
  - [ ] Permutations, combinations, etc. (optimized with gcd)
  - [ ] Larger number types
- Numerical methods/algebra
  - [ ] Tensors
  - [ ] Random number generator for arrays and scalars
  - [ ] Linear algebra methods
  - [ ] A method like NumPy's `linspace`
  - [ ] Numerical approximations (for things like derivatives and integrals)
- Misc
  - [ ] Lua API
  - [ ] Get rid of the unfathomable amount of `.unwrap()`
  - [ ] Make better errors
  - [ ] My own CSV reading
