use std::time::Instant;

use metalearning::trainer::NetworkTrainer;
use metalearning::{helper::vec_total_diff, network::Network};
use metalearning::{Datapair, Dataset};
use rust_mnist::Mnist;

#[allow(dead_code)]
fn display_number(image: &Vec<f64>) {
    for y in 0..28 {
        for x in 0..28 {
            print!(
                "{}",
                match (image[y * 28 + x] * 100.0) as u32 {
                    0..=50 => "  ",
                    _ => "XX",
                }
            )
        }
        println!();
    }
}
#[allow(dead_code)]
fn parse_output(v: &Vec<f64>) -> (usize, f64) {
    let mut max = (0, 0.0);
    for (index, value) in v.iter().enumerate() {
        if *value > max.1 {
            max = (index, *value);
        }
    }
    return max;
}

fn main() {
    println!("Loading some mnist stuff...");
    let mnist = Mnist::new("data/");
    let data = mnist
        .train_data
        .iter()
        .map(|i| {
            i.iter()
                .map(|x| ((x + 0) as f64 / 255.0) as f64)
                .collect::<Vec<_>>()
        })
        .zip(mnist.train_labels.iter().map(|c| {
            let mut v = vec![];
            for i in 0..10 as u8 {
                if i == *c {
                    v.push(1.0);
                } else {
                    v.push(0.0);
                }
            }
            return v;
        }))
        .map(|(input, output)| Datapair(input, output))
        .collect::<Vec<_>>();

    let mut t = NetworkTrainer {
        dataset: Dataset(data),
        networks: vec![Network::new(vec![20, 15, 10 as usize], (28 * 28) as usize)],
        population: 50,
        decimation_count: 45,
        accuracy_samples: 5,
        mutation: 0.05,
        verbose: true,
        accuracy: 0.0,
    };
    println!("training started!");
    loop {
        let start = Instant::now();
        t.train(100);
        let duration = start.elapsed();
        println!("Time for training 100 iterations: {:?}", duration);
        let test = t.dataset.get_n(1)[0];
        let o = t.networks[0].eval(&test.0);
        display_number(&test.0);
        println!("The stupid network reported values: {:?}", parse_output(&o));
        println!("Correct would be: {:?}", parse_output(&test.1));
        println!("Thats a accuracy of {}\n", vec_total_diff(&o, &test.1))
    }
}
