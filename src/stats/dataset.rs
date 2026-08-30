use std::fs;

use crate::stats::{Value, column::Column};

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

    #[error(
        "This dataset contains rows which are inconsistent with what this method expects. Try encoding or removing the column(s)."
    )]
    InconsistentColumns,

    #[error("There are no columns,")]
    NoColumns,
}

pub type DatasetResult<T> = Result<T, DatasetError>;

#[derive(Clone, Debug)]
pub struct Dataset {
    columns: Vec<(String, Column)>,
}

// TODO: make a macro for creating Datasets inline

// abstraction methods for the inner Vec<(String, Column)> representation
impl Dataset {
    fn inner_contains(&self, key: &str) -> bool {
        for (name, _) in &self.columns {
            if name.as_str() == key {
                return true;
            }
        }

        false
    }

    fn inner_remove(&mut self, key: &str) {
        for (i, (name, _)) in self.columns.iter().enumerate() {
            if name.as_str() == key {
                self.columns.remove(i);
                return;
            }
        }
    }

    fn inner_get(&self, key: &str) -> DatasetResult<&(String, Column)> {
        for entry in &self.columns {
            if entry.0.as_str() == key {
                return Ok(entry);
            }
        }

        Err(DatasetError::IndexInexistent)
    }

    fn inner_get_mut(&mut self, key: &str) -> DatasetResult<&mut (String, Column)> {
        for entry in &mut self.columns {
            if entry.0.as_str() == key {
                return Ok(entry);
            }
        }

        Err(DatasetError::IndexInexistent)
    }
}

impl Dataset {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.columns.len()
    }

    // returns copy
    pub fn drop(self, column_names: &[&str]) -> DatasetResult<Self> {
        // skip the case with no columns to drop
        if column_names.len() == 0 {
            return Ok(self);
        }

        let mut new_ds = self.clone();
        for name in column_names {
            if !new_ds.inner_contains(*name) {
                return Err(DatasetError::IndexInexistent);
            }

            new_ds.inner_remove(*name);
        }

        Ok(new_ds)
    }

    // prints type and length and shit
    pub fn column_info(&self) -> String {
        let mut info = String::new();

        for (key, col) in &self.columns {
            match col {
                Column::Int(v) => {
                    info += format!("Name: {}, Type: Int, Length: {}\n", key, v.len()).as_str()
                }
                Column::Float(v) => {
                    info += format!("Name: {}, Type: Float, Length: {}\n", key, v.len()).as_str()
                }
                Column::String(v) => {
                    info += format!("Name: {}, Type: String, Length: {}\n", key, v.len()).as_str()
                }
            }
        }

        info
    }

    pub fn add_column(&mut self, key: String, col: Column) -> DatasetResult<()> {
        if self.inner_contains(&key) {
            Err(DatasetError::IndexInUse)
        } else {
            self.columns.push((key, col));
            Ok(())
        }
    }

    // get the hashmap of columns
    pub fn get_columns(&self) -> &Vec<(String, Column)> {
        &self.columns
    }

    // get the rows as a list of f64's, which is what models want
    // WARNING: please optimize this, its ass
    // TODO: (maybe) return a flat array
    pub fn get_rows_as_vec_f64(&self) -> DatasetResult<Vec<Vec<f64>>> {
        if self.columns.is_empty() {
            return Err(DatasetError::NoColumns);
        }

        let mut vec: Vec<Vec<f64>> = Vec::with_capacity(self.len());

        for (_, col) in &self.columns {
            match col {
                Column::Int(v) => vec.push(v.iter().map(|&el| el as f64).collect()),
                Column::Float(v) => vec.push(v.clone()),
                Column::String(_) => return Err(DatasetError::InconsistentColumns),
            }
        }

        // transpose vector
        let mut transpose: Vec<Vec<f64>> = Vec::with_capacity(vec[0].len());
        for i in 0..vec[0].len() {
            let mut row = Vec::with_capacity(vec.len());

            for j in 0..vec.len() {
                row.push(vec[j][i]);
            }

            transpose.push(row);
        }

        Ok(transpose)
    }

    pub fn get_columns_as_vec_f64(&self) -> DatasetResult<Vec<Vec<f64>>> {
        if self.columns.is_empty() {
            return Err(DatasetError::NoColumns);
        }

        let mut vec: Vec<Vec<f64>> = Vec::with_capacity(self.len());

        for (_, col) in &self.columns {
            match col {
                Column::Int(v) => vec.push(v.iter().map(|&el| el as f64).collect()),
                Column::Float(v) => vec.push(v.clone()),
                Column::String(_) => return Err(DatasetError::InconsistentColumns),
            }
        }

        Ok(vec)
    }

    // get immutable reference to a column
    pub fn get_column(&self, key: &str) -> DatasetResult<&Column> {
        self.inner_get(key).map(|entry| &entry.1)
    }

    // get mutable reference to column
    pub fn get_column_mut(&mut self, key: &str) -> DatasetResult<&mut Column> {
        self.inner_get_mut(key).map(|entry| &mut entry.1)
    }

    // NOTE: this makes operations not in place
    pub fn get_row(&self, idx: usize) -> DatasetResult<Vec<Value>> {
        let mut row: Vec<Value> = Vec::with_capacity(self.columns.len());

        for (_, col_data) in &self.columns {
            match col_data {
                Column::Float(v) => row.push(Value::Float(v[idx])),
                Column::Int(v) => row.push(Value::Int(v[idx])),
                Column::String(v) => row.push(Value::String(v[idx].clone())),
            }
        }

        Ok(row)
    }

    pub fn read_string_csv(data: String) -> DatasetResult<Dataset> {
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

            // TODO: make this cleaner, maybe try converting first and continue if its Ok()

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

            if i64_vec.len() == n_rows {
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

            if f64_vec.len() == n_rows {
                ds.add_column(headers[i].clone(), Column::Float(f64_vec))?;
                continue;
            }

            // if it fails now, it must be a string
            ds.add_column(headers[i].clone(), Column::String(col_elements))?;
        }

        Ok(ds)
    }

    pub fn read_csv(path: &str) -> DatasetResult<Dataset> {
        let buf: String = fs::read_to_string(path).map_err(|_| DatasetError::FileFailure)?;
        Dataset::read_string_csv(buf)
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
        let ds = Dataset::read_string_csv(String::from("a,b,c\n1,2,3\n4,5,6")).unwrap();

        for (col_name, col_data) in ds.get_columns() {
            println!("{col_name}: {:?}", col_data);
        }
    }

    #[test]
    #[should_panic]
    fn csv_failure() {
        // unterminated string entry
        let _ = Dataset::read_string_csv(String::from("a,b\n\"unterminated")).unwrap();

        // duplicate column
        let _ = Dataset::read_string_csv(String::from("a,b,b\n0,1,1")).unwrap();
    }
}
