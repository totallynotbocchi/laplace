use std::{
    collections::{
        HashMap,
        hash_map::Entry::{Occupied, Vacant},
    },
    fmt::Display,
};

use crate::stats::column::Column;

// error type
#[derive(Debug, PartialEq, PartialOrd)]
pub enum DatasetError {
    IndexInUse,
    IndexInexistent,
}

impl Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg: &str = match self {
            Self::IndexInUse => "This index is already being used.",
            Self::IndexInexistent => "This index does not exist.",
        };

        write!(f, "Dataset Error: {}", msg)
    }
}

pub struct Dataset {
    data: HashMap<&'static str, Column>,
}

impl Dataset {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn add_column(&mut self, key: &'static str, col: Column) -> Result<(), DatasetError> {
        match self.data.entry(key) {
            Vacant(entry) => {
                entry.insert(col);
                Ok(())
            }
            Occupied(_) => Err(DatasetError::IndexInUse),
        }
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
        match self.data.entry(key) {
            // we use .into_mut rather than .get_mut because it takes ownership of the entry, and
            // it can safely return a mutable reference to the column now
            Occupied(entry) => Ok(entry.into_mut()),
            Vacant(_) => Err(DatasetError::IndexInexistent),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_column() {
        let mut ds = Dataset::new();

        // add a column and check if it was added successfully
        assert_eq!(ds.add_column("Hi", Column::Int(vec![6, 7])), Ok(()));

        // add a column with the same index and check if its an error
        assert_eq!(
            ds.add_column("Hi", Column::Int(vec![6, 7, 8])),
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
}
