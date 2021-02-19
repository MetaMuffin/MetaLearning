use rand::Rng;

pub fn random_vec(size: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut v: Vec<f64> = Vec::with_capacity(size);
    for i in 0..v.len() {
        v[i] = rng.gen();
    }
    return v;
}

pub fn zero_vec(size: usize) -> Vec<f64> {
    let mut v: Vec<f64> = Vec::with_capacity(size);
    for i in 0..v.len() {
        v[i] = 0.0;
    }
    return v;
}

pub fn fast_sigmoid(v: f64) -> f64 {
    return v / (1.0 + v.abs());
}
