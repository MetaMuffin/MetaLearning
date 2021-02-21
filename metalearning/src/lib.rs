use rand::Rng;

pub mod helper;
pub mod network;
pub mod trainer;

pub struct Datapair(pub Vec<f64>, pub Vec<f64>);
pub struct Dataset(pub Vec<Datapair>);

impl Dataset {
    pub fn get_n<'a>(&'a self, count: usize) -> Vec<&'a Datapair> {
        let mut rng = rand::thread_rng();
        let mut out = vec![];
        for _ in 0..count {
            let index = rng.gen_range(0..self.0.len());
            out.push(&self.0[index]);
        }

        return out;
    }
}
