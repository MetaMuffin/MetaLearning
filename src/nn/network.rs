use super::{Datapair, helper::{fast_sigmoid, random_vec, vec_total_diff, zero_vec}};
use rand::Rng;
// A Feed-forward neural network
#[derive(Debug, Clone)]
pub struct Network {
    biases: Vec<Vec<f64>>,
    weights: Vec<Vec<Vec<f64>>>,
    size: Vec<usize>
}


impl Network {
    pub fn new(layers: Vec<usize>, input_size: usize) -> Network {
        let mut biases = vec![];
        let mut weights = vec![];

        for (li, lsize) in layers.iter().enumerate() {
            biases.push(zero_vec(*lsize));
            weights.push(Vec::new());
            for ni in 0..*lsize {
                let mut pnsize = input_size;
                if ni > 0 {
                    pnsize = layers[ni - 1];
                }
                weights[li].push(random_vec(pnsize));
            }
        }

        Network {
            biases,
            weights,
            size: layers,
        }
    }

    fn eval_layer(&self, li: usize, last: &Vec<f64>) -> Vec<f64> {
        let mut out = vec![];
        for ti in 0..self.size[li] {
            let mut v = 0.0;
            for si in 0..last.len() {
                let weight = self.weights[li][ti][si];
                let in_val = last[si];
                let weighted = in_val * weight;
                v += weighted;
            }
            let bias = self.biases[li][ti];
            let value = fast_sigmoid(bias + v);
            out.push(value);
        }
        return out;
    }

    pub fn eval(&self, input: &Vec<f64>) -> Vec<f64> {
        let mut prev = None;
        for li in 0..self.biases.len() {
            prev = Some(match prev {
                Some(p) => self.eval_layer(li, &p),
                None => self.eval_layer(li, input),
            });
        }
        return prev.unwrap();
    }

    pub fn accuracy(&self, datapair: &Datapair) -> f64 {
        let output = self.eval(&datapair.0);
        return vec_total_diff(&output, &datapair.1);
    }
    pub fn accuracy_set(&self, dataset: &[&Datapair]) -> f64 {
        let mut a = 0.0;
        for p in dataset {
            a += self.accuracy(p);
        }
        return a;
    }

    pub fn mutate(&mut self, factor: f64) {
        let mut rng = rand::thread_rng();
        self.biases = self
            .biases
            .iter()
            .map(|s| {
                s.iter()
                    .map(|x: &f64| *x + rng.gen::<f64>() * factor)
                    .collect()
            })
            .collect();
        self.weights = self
            .weights
            .iter()
            .map(|s| {
                s.iter()
                    .map(|t| {
                        t.iter()
                            .map(|x: &f64| *x + rng.gen::<f64>() * factor)
                            .collect()
                    })
                    .collect()
            })
            .collect();
    }
}
