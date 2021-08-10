use std::f32::consts::SQRT_2;

use once_cell::sync::Lazy;

use test_utils::ltr::model;

use super::{ndlayers::ActivationFunction, *};

mod inference;
mod training;

static LIST_NET: Lazy<ListNet> =
    Lazy::new(|| ListNet::deserialize_from_file(model().unwrap()).unwrap());

#[test]
fn test_chunk_size_is_valid() {
    assert_eq!(ListNet::CHUNK_SIZE * 2, ListNet::INPUT_NR_DOCUMENTS);
}

#[test]
fn test_random_weights_initialization() {
    let ListNet {
        dense1,
        dense2,
        scores,
        prob_dist,
    } = ListNet::new_with_random_weights();

    test_layer(&dense1);
    test_layer(&dense2);
    test_layer(&scores);
    test_layer(&prob_dist);

    fn test_layer(layer: &Dense<impl ActivationFunction<f32>>) {
        for b in layer.bias().iter() {
            assert_approx_eq!(f32, b, 0.0, ulps = 9)
        }
        let weights = layer.weights();
        let std = SQRT_2 / (weights.shape()[0] as f32).sqrt();
        let limit = 2. * std;
        for &w in weights.iter() {
            assert!(
                -limit <= w && w <= limit,
                "out of bound weight: {} <= {} <= {}",
                -limit,
                w,
                limit
            );
        }
    }
}

#[test]
fn test_serialize_deserialize_list_net() {
    let list_net = ListNet::new_with_random_weights();
    let mut buffer = Vec::new();
    list_net.clone().serialize_into(&mut buffer).unwrap();
    let list_net2 = ListNet::deserialize_from(&*buffer).unwrap();
    assert_approx_eq!(f32, list_net.dense1.weights(), list_net2.dense1.weights());
    assert_approx_eq!(f32, list_net.dense1.bias(), list_net2.dense1.bias());
    assert_approx_eq!(f32, list_net.dense2.weights(), list_net2.dense2.weights());
    assert_approx_eq!(f32, list_net.dense2.bias(), list_net2.dense2.bias());
    assert_approx_eq!(f32, list_net.scores.weights(), list_net2.scores.weights());
    assert_approx_eq!(f32, list_net.scores.bias(), list_net2.scores.bias());
    assert_approx_eq!(
        f32,
        list_net.prob_dist.weights(),
        list_net2.prob_dist.weights()
    );
    assert_approx_eq!(f32, list_net.prob_dist.bias(), list_net2.prob_dist.bias());
}
