use super::helper::{fast_sigmoid, random_vec,zero_vec};

// A Feed-forward neural network
#[derive(Debug, Clone)]
pub struct Network {
    biases: Vec<Vec<f64>>,
    weights: Vec<Vec<Vec<f64>>>,
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
                    pnsize = biases[ni - 1].len();
                }
                weights[li].push(random_vec(pnsize));
            }
        }

        Network { biases, weights }
    }

    fn eval_layer(&self, li: usize, last: Vec<f64>) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.biases[li].len());
        for ti in 0..out.len() {
            let mut v = 0.0;
            for si in 0..last.len() {
                let weight = self.weights[li][ti][si];
                let in_val = last[si];
                let weighted = in_val * weight;
                v += weighted;
            }
            let bias = self.biases[li][ti];
            let value = fast_sigmoid(bias + v);
            out[ti] = value
        }
        return out;
    }

    pub fn eval(&self, inputs: Vec<f64>) -> Vec<f64> {
        let mut prev = inputs;
        for li in 0..self.biases.len() {
            prev = self.eval_layer(li, prev);
        }
        return prev;
    }
}
