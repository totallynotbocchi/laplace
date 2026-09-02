use std::{
    iter::Sum,
    ops::{Add, Mul, Sub},
};

use num_traits::{AsPrimitive, Num};
use thiserror::Error;

use crate::algebra::Vector;

#[derive(Debug, Error, Clone, Copy)]
pub enum MatrixError {
    #[error("The matrices' dimensions are not the same, when they were ecpected to be.")]
    MismatchingDimensions,
}

type MatrixResult<T> = Result<T, MatrixError>;

// matrix type
#[derive(Debug, Clone)]
pub struct Matrix<T, const ROWS: usize, const COLS: usize> {
    data: [[T; COLS]; ROWS], // row major
}

// default matrix is the zero matrix
impl<T, const ROWS: usize, const COLS: usize> Default for Matrix<T, ROWS, COLS>
where
    T: Num + Copy,
{
    fn default() -> Self {
        Self {
            data: [[T::zero(); COLS]; ROWS],
        }
    }
}

// create matrix from nested array
impl<T, const ROWS: usize, const COLS: usize> From<[[T; COLS]; ROWS]> for Matrix<T, ROWS, COLS> {
    fn from(value: [[T; COLS]; ROWS]) -> Self {
        Self { data: value }
    }
}

// get rows
impl<T, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS>
where
    T: Clone + Copy + Default,
{
    // clone row
    pub fn get_row(&self, idx: usize) -> MatrixResult<Vector<T, COLS>> {
        if idx >= ROWS {
            return Err(MatrixError::MismatchingDimensions);
        }

        let raw_row = self.data[idx].clone();
        let mut row: [T; COLS] = [T::default(); COLS];

        for j in 0..COLS {
            row[j] = raw_row[j].clone();
        }

        Ok(Vector::from(row))
    }

    // clone column
    pub fn get_column(&self, idx: usize) -> MatrixResult<Vector<T, ROWS>> {
        if idx >= COLS {
            return Err(MatrixError::MismatchingDimensions);
        }

        let mut col: [T; ROWS] = [T::default(); ROWS];
        for i in 0..ROWS {
            col[i] = self.data[i][idx];
        }

        Ok(Vector::from(col))
    }
}

// basic matrix methods for any type and columns
impl<T, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS>
where
    T: Clone + Copy,
{
    pub fn set_row(&mut self, i: usize, row: Vector<T, COLS>) {
        self.data[i] = row.into();
    }

    // immutable access
    pub fn at(&self, i: usize, j: usize) -> MatrixResult<&T> {
        if i >= ROWS || j >= COLS {
            Err(MatrixError::MismatchingDimensions)
        } else {
            Ok(&self.data[i][j])
        }
    }

    // mutable access
    pub fn at_mut(&mut self, i: usize, j: usize) -> MatrixResult<&mut T> {
        if i >= ROWS || j >= COLS {
            Err(MatrixError::MismatchingDimensions)
        } else {
            Ok(&mut self.data[i][j])
        }
    }

    pub fn set(&mut self, i: usize, j: usize, value: T) {
        self.data[i][j] = value;
    }

    pub fn shape(&self) -> (usize, usize) {
        (ROWS, COLS)
    }
}

// cloning accessor
impl<T, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS>
where
    T: Clone,
{
    // clone the value
    pub fn at_clone(&self, i: usize, j: usize) -> MatrixResult<T> {
        if i >= ROWS || j >= COLS {
            Err(MatrixError::MismatchingDimensions)
        } else {
            Ok(self.data[i][j].clone())
        }
    }
}

// matrix equals
impl<T, const ROWS: usize, const COLS: usize> PartialEq for Matrix<T, ROWS, COLS>
where
    T: PartialEq + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        for i in 0..ROWS {
            for j in 0..COLS {
                if self.at_clone(i, j).unwrap() != other.at_clone(i, j).unwrap() {
                    return false;
                }
            }
        }

        true
    }

    fn ne(&self, other: &Self) -> bool {
        for i in 0..ROWS {
            for j in 0..COLS {
                if self.at_clone(i, j).unwrap() == other.at_clone(i, j).unwrap() {
                    return false;
                }
            }
        }

        true
    }
}

// matrix addition
impl<T, const ROWS: usize, const COLS: usize> Add for Matrix<T, ROWS, COLS>
where
    T: Num + Clone + Copy,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut new_mat = self.clone();

        for i in 0..ROWS {
            for j in 0..COLS {
                let sum = self.at_clone(i, j).unwrap() + rhs.at_clone(i, j).unwrap();
                new_mat.set(i, j, sum);
            }
        }

        new_mat
    }
}

// matrix scalar multiplication
impl<T, const ROWS: usize, const COLS: usize> Mul<T> for Matrix<T, ROWS, COLS>
where
    T: Num + Clone + Copy,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut new_mat = self.clone();

        for i in 0..ROWS {
            for j in 0..COLS {
                let scaled = self.at_clone(i, j).unwrap() * rhs;
                new_mat.set(i, j, scaled);
            }
        }

        new_mat
    }
}

// matrix multiplication
// AxB * B*C = AxC
impl<T, const A: usize, const B: usize, const C: usize> Mul<Matrix<T, B, C>> for Matrix<T, A, B>
where
    T: Num + Clone + Copy + Default + Sum,
{
    type Output = Matrix<T, A, C>;

    fn mul(self, rhs: Matrix<T, B, C>) -> Self::Output {
        let mut new_mat: Matrix<T, A, C> = Matrix::default();

        for i in 0..A {
            for j in 0..B {
                let row = self.get_row(i).unwrap();
                let col = rhs.get_column(j).unwrap();

                new_mat.set(i, j, row * col);
            }
        }

        new_mat
    }
}

// matrix subtraction
impl<T, const ROWS: usize, const COLS: usize> Sub for Matrix<T, ROWS, COLS>
where
    T: Num + Clone + Copy,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut new_mat = self.clone();

        for i in 0..ROWS {
            for j in 0..COLS {
                let sum = self.at_clone(i, j).unwrap() - rhs.at_clone(i, j).unwrap();
                new_mat.set(i, j, sum);
            }
        }

        new_mat
    }
}

// TODO: matrix multiplication....

// matrix determiant (only for NxN)
impl<T, const N: usize> Matrix<T, N, N>
where
    T: Num + Copy + Mul<T> + PartialOrd + Default + AsPrimitive<f64>,
{
    // get the determinant of a matrix
    fn det(&self) -> f64 {
        let mut det = 1.;

        // clone the elimination matrix as f64
        let mut elim_mat = Matrix::<f64, N, N>::default();
        for i in 0..N {
            for j in 0..N {
                elim_mat.set(j, i, self.at_clone(j, i).unwrap().as_());
            }
        }

        // do gaussian elimination
        for i in 0..N {
            let mut pivot_row = i;
            let mut pivot_entry = elim_mat.at_clone(i, i).unwrap();

            // find the biggest value in this pivot column, so we go down in rows
            // NOTE: we use the maximum to avoid floating point errors
            let mut max = pivot_entry;
            for j in (i + 1)..N {
                let entry_below = elim_mat.at_clone(j, i).unwrap();

                if entry_below.abs() > max.abs() {
                    max = entry_below;
                    pivot_row = j;
                }
            }

            // if the maximum value's pivot isnt the same as before, swap the rows for easier
            // operations, so pivot_row continues to be i
            if pivot_row != i {
                let old_row = elim_mat.get_row(pivot_row).unwrap();
                let new_row = elim_mat.get_row(i).unwrap();

                elim_mat.set_row(pivot_row, new_row);
                elim_mat.set_row(i, old_row);

                // update the current pivot entry
                pivot_entry = elim_mat.at_clone(i, i).unwrap();

                // swap det sign becaude we swapped rows
                det *= -1.;
            }

            // if this pivot zero, theres no valid pivot, so we have a det of 0
            if pivot_entry == 0. {
                return 0.;
            }

            // continue the elimination process.

            // eliminate every entry below pivot_row
            for j in i + 1..N {
                // start with the upper most one
                let entry_below = elim_mat.at_clone(j, i).unwrap();
                let factor = entry_below / pivot_entry;

                // go thru every column, which is in each row below the pivot
                for k in i..N {
                    let entry = elim_mat.at_clone(j, k).unwrap();
                    let pivot_row_entry = elim_mat.at_clone(i, k).unwrap();

                    elim_mat.set(j, k, entry - factor * pivot_row_entry);
                }
            }

            det *= pivot_entry;
        }

        det
    }
}

// matrix transpose
impl<T, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS>
where
    T: Num + Clone + Copy + Default,
{
    fn transpose(&self) -> Matrix<T, COLS, ROWS> {
        let mut new_mat: Matrix<T, COLS, ROWS> = Matrix::default();

        for i in 0..ROWS {
            for j in 0..COLS {
                new_mat.set(j, i, self.at_clone(i, j).unwrap());
            }
        }

        new_mat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_2_matrix_det() {
        let mat = Matrix::from([[1, 2], [3, 4]]);

        // det A = 1(4) - 2(3) = 4 - 6 = -2
        assert_eq!(mat.det(), -2.);
    }

    #[test]
    fn transpose() {
        let mat = Matrix::from([[1, 2], [3, 4]]);

        assert_eq!(mat.transpose(), Matrix::from([[1, 3], [2, 4]]));
    }

    #[test]
    fn matrix_mult() {
        let a = Matrix::from([[1, 2], [3, 4]]);
        let b = Matrix::from([[5, 6], [7, 8]]);

        assert_eq!(a * b, Matrix::from([[19, 22], [43, 50]]))
    }
}
