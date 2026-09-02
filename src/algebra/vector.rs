use std::{
    iter::Sum,
    ops::{Add, Mul, Neg, Sub},
};

use num_traits::{Num, ToPrimitive};

// struct for n-dimensional vectors
#[derive(Debug)]
pub struct Vector<T, const N: usize> {
    data: [T; N],
}

// turn array into Vector
impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(value: [T; N]) -> Self {
        Self { data: value }
    }
}

impl<T, const N: usize> Into<[T; N]> for Vector<T, N>
where
    T: Clone,
{
    fn into(self) -> [T; N] {
        self.data.clone()
    }
}

// default value is the zero vector
impl<T, const N: usize> Default for Vector<T, N>
where
    T: Num + Copy,
{
    fn default() -> Self {
        Self {
            data: [T::zero(); N],
        }
    }
}

// dot product
impl<T, const N: usize> Mul<Vector<T, N>> for Vector<T, N>
where
    T: Num + Copy + Sum + Mul<T>,
{
    type Output = T;

    fn mul(self, rhs: Vector<T, N>) -> Self::Output {
        self.data.iter().zip(rhs.data).map(|(&a, b)| a * b).sum()
    }
}

// vector addition
impl<T, const N: usize> Add for Vector<T, N>
where
    T: Num + ToPrimitive + Copy + Add<T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut v: [T; N] = self.data.clone();
        v.iter_mut().zip(rhs.data).for_each(|(x, y)| *x = *x + y);

        Self { data: v }
    }
}

// vector subtraction
impl<T, const N: usize> Sub for Vector<T, N>
where
    T: Num + Copy + Sub<T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut v: [T; N] = self.data.clone();
        v.iter_mut().zip(rhs.data).for_each(|(x, y)| *x = *x - y);

        Self { data: v }
    }
}

// vector negation (negates all components)
impl<T, const N: usize> Neg for Vector<T, N>
where
    T: Num + Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            data: self.data.map(|x| -x),
        }
    }
}

// make double equals work as expected
impl<T, const N: usize> PartialEq for Vector<T, N>
where
    T: Num + ToPrimitive + Copy + Add<T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data).all(|(&x, y)| x == y)
    }

    fn ne(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data).all(|(&x, y)| x != y)
    }
}

// scalar multiplication
impl<T, const N: usize> Mul<T> for Vector<T, N>
where
    T: Num + Copy + Add<T> + Sum + std::fmt::Debug,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            data: self
                .data
                .iter()
                .map(|x| (*x) * rhs)
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        }
    }
}

// general methods
impl<T, const N: usize> Vector<T, N> {
    // immutable reference
    pub fn at(&self, i: usize) -> Option<&T> {
        if i >= N {
            return None;
        }

        Some(&self.data[i])
    }

    // mutable reference
    pub fn at_mut(&mut self, i: usize) -> Option<&mut T> {
        if i >= N {
            return None;
        }

        Some(&mut self.data[i])
    }
}

// methods for T that implement Clone
impl<T, const N: usize> Vector<T, N>
where
    T: Clone,
{
    // copy getter
    pub fn at_clone(&self, i: usize) -> Option<T> {
        if i >= N {
            return None;
        }

        Some(self.data[i].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum() {
        let v1 = Vector::from([1, 2, 3]);
        let v2 = Vector::from([1, 2, 3]);

        assert_eq!(v1 + v2, Vector::from([2, 4, 6]));
    }

    #[test]
    fn scalar_mult() {
        let v = Vector::from([1, 2, 3]);
        assert_eq!(v * 3, Vector::from([3, 6, 9]));
    }

    #[test]
    fn dot_product() {
        let v1 = Vector::from([1, 2, 3]);
        let v2 = Vector::from([1, 2, 3]);

        assert_eq!(v1 * v2, 14);
    }
}
