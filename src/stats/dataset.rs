use std::{
    collections::{
        HashMap,
        hash_map::Entry::{Occupied, Vacant},
    },
    fs::{self, File},
    io::Read,
};

use crate::stats::column::Column;

use thiserror::Error;

// error type
#[derive(Error, Debug, PartialEq, PartialOrd)]
pub enum DatasetError {
    #[error("This index is already being used.")]
    IndexInUse,

    #[error("This index does not exist.")]
    IndexInexistent,

    #[error(
        "Something went wrong when reading the file. Check if the exact path is correct and if you have the neccesary permissions."
    )]
    FileFailure,

    #[error("The CSV data is malformed, empty or invalid. Check the file's contents.")]
    CSVError,
}

pub struct Dataset {
    data: HashMap<String, Column>,
}

impl Dataset {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn add_column(&mut self, key: String, col: Column) -> Result<(), DatasetError> {
        match self.data.entry(key.to_string()) {
            Vacant(entry) => {
                entry.insert(col);
                Ok(())
            }
            Occupied(_) => Err(DatasetError::IndexInUse),
        }
    }

    pub fn get_columns(&self) -> &HashMap<String, Column> {
        &self.data
    }

    // get immutable reference to column
    pub fn get_column(&self, key: &'static str) -> Result<&Column, DatasetError> {
        match self.data.get(key) {
            Some(value) => Ok(value),
            None => Err(DatasetError::IndexInexistent),
        }
    }

    // get mutable reference to column
    pub fn get_column_mut(&mut self, key: &'static str) -> Result<&mut Column, DatasetError> {
        match self.data.entry(key.to_string()) {
            // we use .into_mut rather than .get_mut because it takes ownership of the entry, and
            // it can safely return a mutable reference to the column now
            Occupied(entry) => Ok(entry.into_mut()),
            Vacant(_) => Err(DatasetError::IndexInexistent),
        }
    }

    pub fn from_string(data: String) -> Result<Dataset, DatasetError> {
        let mut rdr = csv::Reader::from_reader(data.as_bytes());

        // define dataset and data
        let mut ds = Dataset::new();

        // get the headers from row 0
        let headers: Vec<String> = rdr
            .headers()
            .map_err(|_| DatasetError::CSVError)?
            .iter()
            .map(|s| s.to_string())
            .collect();

        // get every value below row 0
        let mut rows: Vec<Vec<String>> = rdr
            .records()
            .map(|r| r.map(|rec| rec.iter().map(|v| v.to_string()).collect()))
            .collect::<Result<_, _>>()
            .map_err(|_| DatasetError::CSVError)?;

        // loop thru every *column*
        let n_cols = headers.len();
        let n_rows = rows.len();

        for i in 0..n_cols {
            // make a list of column elements (for Column)
            let mut col_elements: Vec<String> = Vec::with_capacity(n_rows);

            // loop thru every row
            for j in 0..n_rows {
                // transfer ownership and put the element into the column
                col_elements.push(std::mem::take(&mut rows[j][i]));
            }

            // try converting into i64
            let mut i64_vec: Vec<i64> = Vec::with_capacity(n_rows);
            for el in &col_elements {
                match el.parse::<i64>() {
                    Ok(num) => i64_vec.push(num),
                    Err(_) => {}
                }
            }

            if (&i64_vec).len() == n_rows {
                ds.add_column(headers[i].clone(), Column::Int(i64_vec))?;
                continue;
            }

            // try converting into f64
            let mut f64_vec: Vec<f64> = Vec::with_capacity(n_rows);
            for el in &col_elements {
                match el.parse::<f64>() {
                    Ok(num) => f64_vec.push(num),
                    Err(_) => {}
                }
            }

            if (&f64_vec).len() == n_rows {
                ds.add_column(headers[i].clone(), Column::Float(f64_vec))?;
                continue;
            }

            // if it fails now, it must be a string
            ds.add_column(headers[i].clone(), Column::String(col_elements))?;
        }

        Ok(ds)
    }

    pub fn read_csv(path: &'static str) -> Result<Dataset, DatasetError> {
        let buf: String = fs::read_to_string(path).map_err(|_| DatasetError::IndexInUse)?;
        Dataset::from_string(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_column() {
        let mut ds = Dataset::new();

        // add a column and check if it was added successfully
        assert_eq!(
            ds.add_column("Hi".to_string(), Column::Int(vec![6, 7])),
            Ok(())
        );

        // add a column with the same index and check if its an error
        assert_eq!(
            ds.add_column("Hi".to_string(), Column::Int(vec![6, 7, 8])),
            Err(DatasetError::IndexInUse)
        );

        // check the column's value
        assert_eq!(ds.get_column("Hi"), Ok(&Column::Int(vec![6, 7])));

        // mutate column and check new value
        match ds.get_column_mut("Hi") {
            Ok(Column::Int(v)) => v.push(8),
            _ => panic!("Getting mutable ref failed."),
        }
        assert_eq!(ds.get_column("Hi"), Ok(&Column::Int(vec![6, 7, 8])));
    }

    #[test]
    // this one requires "cargo test csv -- --nocapture"
    fn csv() {
        let ds = Dataset::from_string(String::from("a,b,c\n1,2,3\n4,5,6")).unwrap();

        for (col_name, col_data) in ds.get_columns() {
            println!("{col_name}: {:?}", col_data);
        }
    }

    #[test]
    #[should_panic]
    fn csv_failure() {
        let _ = Dataset::from_string(String::from("a,b\n\"unterminated")).unwrap();
    }
}
