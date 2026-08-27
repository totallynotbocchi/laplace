use laplace::{
    models::{LinearRegression, Model},
    stats::Dataset,
};

fn main() {
    // load
    let ds = Dataset::read_csv("./examples/Student_Performance.csv")
        .unwrap()
        .drop(&["Extracurricular Activities"]) // its non-numeric
        .unwrap();

    // print info
    println!("{}", ds.column_info());

    // split variables
    let y = ds
        .get_column("Performance Index")
        .unwrap()
        .as_f64_vec()
        .unwrap();

    let x = ds
        .drop(&["Performance Index"])
        .unwrap()
        .get_rows_as_vec_f64()
        .unwrap();

    println!(
        "X columns and length: ({}, {}), y len: {}",
        x.len(),
        x[0].len(),
        y.len()
    );

    // train linear model
    let mut model = LinearRegression::new(500, 1e-4);
    model.fit(&x, &y).unwrap();

    // predict one entry (the first in the csv)
    let pred = model.predict(&vec![7., 99., 9., 1.]).unwrap();
    println!("Prediction is: {pred}");
}
