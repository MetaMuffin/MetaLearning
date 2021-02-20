use rand::Rng;

use super::{helper::float_compareable, network::Network, Dataset};

pub struct NetworkTrainer {
    pub networks: Vec<Network>,
    pub dataset: Dataset,

    pub population: usize,
    pub decimation_count: usize,
    pub accuracy_samples: usize,
    pub mutation: f64,
    pub verbose: bool,
}

impl NetworkTrainer {
    pub fn train(&mut self, iterations: usize) {
        for _ in 0..iterations {
            self.training_iteration();
        }
    }

    pub fn training_iteration(&mut self) {
        self.populate();
        self.decimate();
    }

    pub fn decimate(&mut self) {
        let datapairs = &self.dataset;
        let ac = self.accuracy_samples;
        self.networks.sort_by_cached_key(|e| {
            float_compareable(e.accuracy_set(datapairs.random(ac).as_slice()))
        });
        self.networks.drain(0..self.decimation_count);
    }

    pub fn populate(&mut self) {
        let mut rng = rand::thread_rng();
        while self.networks.len() < self.population {
            let parent = &self.networks[rng.gen_range(0..self.networks.len())];
            let mut new = parent.clone();
            new.mutate(self.mutation);
            self.networks.push(new)
        }
    }
}
