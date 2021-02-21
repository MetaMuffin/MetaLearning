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

    pub accuracy: f64
}

impl NetworkTrainer {
    pub fn train(&mut self, iterations: usize) {
        println!();
        for i in 0..iterations {
            if self.verbose {
                println!(
                    "\x1b[1A\r\x1b[2Ktraining: {}/{} | average accuracy: {:.5}",
                    i,
                    iterations,
                    self.accuracy
                );
            }
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
        let mut sum = 0.0;
        self.networks.sort_by_cached_key(|e| {
            let a = e.accuracy_set(datapairs.get_n(ac).as_slice());
            sum += a;
            float_compareable(a)
        });
        self.accuracy = sum / self.networks.len() as f64;
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
