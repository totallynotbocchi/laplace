pub enum TensorError {
    MismatchingRank,
}

// main datatype
pub struct Tensor<'a, T> {
    data: Vec<T>, // row-major flattened layout
    shape: &'a [usize],
}

impl<'a, T> Tensor<'a, T> {
    pub fn new() -> Self {
        Self {
            data: vec![],
            shape: &[],
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn reshape(mut self, new_shape: &'a [usize]) -> Self {
        self.shape = new_shape;
        self
    }

    fn is_valid(&self, coords: &'a [usize]) -> bool {
        if coords.len() != self.shape.len() {
            return false;
        }

        for i in 0..coords.len() {
            if coords[i] >= self.shape[i] {
                return false;
            }
        }

        true
    }

    fn flat_index(&self, coords: &'a [usize]) -> usize {
        assert!(
            self.is_valid(&coords),
            "The coordinates have a different shape than the tensor.",
        );

        let mut idx: usize = 0;
        for i in 0..self.shape.len() {
            idx = idx * self.shape[i] + coords[i];
        }

        idx
    }

    pub fn get(&self, coords: &'a [usize]) -> &T {
        &self.data[self.flat_index(coords)]
    }

    pub fn get_mut(&'a mut self, coords: &'a [usize]) -> &'a mut T {
        let idx = self.flat_index(coords);
        &mut self.data[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattening() {
        // 2 x 2 matrix
        let ts = Tensor::<i32>::new().reshape(&[3, 3]);

        // check if index (0,1) is flattened correctly.
        // formula is x + W * y
        assert_eq!(ts.flat_index(&[0, 1]), 1);
        assert_eq!(ts.flat_index(&[2, 2]), 8);
    }
}
