use nn::{network::Network, trainer::NetworkTrainer, Datapair, Dataset};

mod nn;

fn main() {
    let mut t = NetworkTrainer {
        dataset: Dataset(vec![
            Datapair(vec![-1.0, -1.0], vec![-1.0]),
            Datapair(vec![1.0, -1.0], vec![1.0]),
            Datapair(vec![-1.0, 1.0], vec![1.0]),
            Datapair(vec![1.0, 1.0], vec![1.0]),
        ]),
        networks: vec![Network::new(vec![2, 1 as usize], 2)],
        population: 50,
        decimation_count: 40,
        accuracy_samples: 4,
        mutation: 0.05,
        verbose: true,
    };

    println!(
        "accuracy before training: {}",
        t.networks[0].accuracy_set(&t.dataset.random(4))
    );

    t.train(100);

    println!(
        "accuracy after training: {}",
        t.networks[0].accuracy_set(&t.dataset.random(4))
    );

    let test_input = vec![-1.0, 1.0];
    println!(
        "input: {:?}\noutput: {:?}",
        test_input,
        t.networks[0].eval(&test_input)
    )
}
