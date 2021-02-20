use std::ops::Mul;

use rand::Rng;

pub fn random_vec(size: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut v = vec![];
    for _ in 0..size {
        v.push(rng.gen());
    }
    return v;
}

pub fn zero_vec(size: usize) -> Vec<f64> {
    let mut v = vec![];
    for _ in 0..size {
        v.push(0.0);
    }
    return v;
}

pub fn fast_sigmoid(v: f64) -> f64 {
    return v / (1.0 + v.abs());
}

pub fn vec_total_diff(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
    let mut w = 0.0;
    for (a1, a2) in v1.iter().zip(v2.iter()) {
        w += (a1 - a2) * (a1 - a2);
    }
    return w;
}

pub fn float_compareable(v: f64) -> i64 {
    return v.mul(1000.0) as i64
}