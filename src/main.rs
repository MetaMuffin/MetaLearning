mod nn;

fn main() {
    let n = nn::network::Network::new(vec![2, 2 as usize], 2);

    let input = vec![1.0, -1.0];
    let output = n.eval(input);
    println!("{:?}", output);
}
