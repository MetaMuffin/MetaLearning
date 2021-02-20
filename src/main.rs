use nn::{network::Network, trainer::NetworkTrainer, Datapair, Dataset};

mod nn;

fn main() {
    let dataset = Dataset(vec![
        Datapair(vec![-1.0, -1.0], vec![-1.0]),
        Datapair(vec![1.0, -1.0], vec![1.0]),
        Datapair(vec![-1.0, 1.0], vec![1.0]),
        Datapair(vec![1.0, 1.0], vec![1.0]),
    ]);

    let mut trainer = NetworkTrainer {
        dataset,
        networks: vec![Network::new(vec![2, 1 as usize], 2)],
        population: 2,
        decimation_count: 1,
        accuracy_samples: 4,
        mutation: 0.1,
        verbose: true,
    };

    trainer.train(2);
}
