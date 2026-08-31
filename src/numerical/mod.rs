mod derivative;
mod integral;

pub use derivative::*;
pub use integral::*;

type RealFn = fn(f64) -> f64;
