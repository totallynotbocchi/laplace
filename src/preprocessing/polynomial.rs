use crate::{
    preprocessing::{DatasetPreprocessor, PreprocessingError, PreprocessingResult},
    stats::{Column, Dataset},
};

pub struct PolynomialTransformation {
    // the vector per exponent of a column
    // index -> highest exponent n (columns are created from 1 to n)
    column_exponents: Vec<i32>,
}

impl PolynomialTransformation {
    pub fn new(column_exponents: &[i32]) -> Self {
        Self {
            column_exponents: Vec::from(column_exponents),
        }
    }
}

impl DatasetPreprocessor for PolynomialTransformation {
    // this method only checks if the number of columns in the data matches
    fn fit(&mut self, data: &Dataset) -> PreprocessingResult<()> {
        if data.len() != self.column_exponents.len() {
            return Err(PreprocessingError::MismatchedColumnCount);
        }

        Ok(())
    }

    fn transform(&self, data: &Dataset) -> PreprocessingResult<Dataset> {
        // get the old ds array
        let old_v = data
            .get_columns_as_vec_f64()
            .map_err(|_| PreprocessingError::MismatchedSizes)?;

        // create new ds
        let mut ds = Dataset::new();

        // for each column
        for (i, col_v) in old_v.iter().enumerate() {
            // for each exponent
            for p in 1..=self.column_exponents[i] {
                // key for the column
                let name = format!("Column {}, exponent {}", i, p);

                // raise each column value to the curent exponent
                let values = col_v.iter().map(|x| x.powi(p as i32)).collect();

                // add it to the new dataset
                ds.add_column(name, Column::Float(values)).unwrap(); // only fails if index is in use
            }
        }

        Ok(ds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic() {
        let mut ds = Dataset::new();
        ds.add_column("1".to_string(), Column::Float(vec![1., 2., 3.]))
            .unwrap();

        let pt = PolynomialTransformation::new(&vec![2]);
        let new_ds = pt.transform(&ds).unwrap();

        let real_v = vec![vec![1., 2., 3.], vec![1., 4., 9.]];
        for ((_, col), real) in new_ds.get_columns().iter().zip(real_v) {
            assert_eq!(col.as_f64_vec().unwrap(), real);
        }
    }
}
