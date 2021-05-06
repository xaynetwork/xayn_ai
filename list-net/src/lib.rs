//! ListNet implementation using NdArray.
//!
//!
//!
//!
//! # Architecture
//!
//! ListNet has a comparable simple architecture, having following
//! stack of layers:
//!
//! 0. Kind Input: Type f32; Shape (nr_docs, nr_of_features);
//! 1. Kind Dense: reLu; 48units; with bias  [nr_of_features => 48]
//! 2. Kind Dense: reLu;  8units; with bias  [48 => 8]
//! 3. Kind Dense: linear; 1units; with bias [8 => 1]
//! 4. Kind Flatten: -- [(nr_docs, 1) => (nr_docs,)]
//! 5. Kind SoftMax: nr_docs units [nr_docs => nr_docs, but sum == 1]

use ndarray::{Array1, Array2, Ix2, LinalgScalar};
use ndutils::nn_layer::{Dense1D, Dense2D};

/**
    input [[features_for_doc_1],
           [features_for_doc_2],
           ...]

    dense == activation(input · weights + bias)
        .e.g reLu(input · weights + bias)


    Dim Transitions (no batch):

    (nr_docs, nr_features) => (nr_docs, 48) => (nr_docs, 8) => (nr_docs => 1) => (nr_docs,) => (nr_docs,)


    Weight Matrices (shapes):
        - (nr_features, 48)
        - (48, 8)
        - (8, 1)  <= we can make this a vector multiplication if we don't use batching for learning (==merged it with flatten)

    Bias Vectors (shapes):
        - (48,)
        - (8,)
        - (1,)


    Activation Functions:
        - linear: (element wise) in keras == identity function
        - reLu: (element wise) max(0, x)
        - softmax: (NOT element wise, but dimensions wise over one axis) o_i = exp(x_i) / sum(over x_j's: exp(x_j))
            - three step implementations: 1) element wise: exp(x) 3) per dimension over axis wise:  sum(x_i) 2) per dimension over axis element wise: x/sum_for_dim
*/

#[macro_use]
mod utils;

pub mod ndutils;

pub struct ListNet<A>
where
    A: LinalgScalar,
{
    dense_1: Dense2D<A>,
    dense_2: Dense2D<A>,
    scores: Dense2D<A>,
    scores_prop_dist: Dense1D<A>,
}

// pub type Data = TODO;

impl<A> ListNet<A>
where
    A: LinalgScalar,
{
    //     pub fn create(&self, nr_documents: usize, nr_features: usize, parameters: Data) -> Self {
    //         //--- all this is independent of nr_documents ---//

    //         // (nr_docs, nr_features) => (nr_docs, 48)
    //         let l_dense_1 = Dense2D::build(nr_features, 48, parameters.get("l_dense_1"), ReLu::new());
    //         // (nr_docs, 48) => (nr_docs, 8)
    //         let l_dense_2 = Dense2D::build(l_dense_1.units, 8, parameters.get("l_dense_2"), ReLu::new());
    //         // (nr_docs, 8) => (nr_docs, 1)
    //         let l_scores = Dense2D::build(l_dense_2.units, 1, parameters.get("l_scores"), Linear::new());

    //         // (nr_docs, 1) => (nr_docs,)
    //         // has not parameters we need to setup

    //         //--- the weights of this depend on nr_documents ---//
    //         // (nr_docs,) => (nr_docs,)
    //         let l_scores_prop_dist = Dense1D::build(nr_documents, parameters.get("l_scores_prop_dist"), SoftMax::new(0));

    //         ListNet {
    //             l_dense_1,
    //             l_dense_2,
    //             l_scores,
    //             l_scores_prop_dist
    //         }
    //     }

    /// Runs List net on the input.
    ///
    /// The input is a 2 dimensional array
    /// with the shape `(number_of_documents, number_of_feature_per_document)`.
    pub fn run(&self, inputs: Array2<A>) -> Array1<A> {
        let dense1_out = self.dense_1.apply_to(inputs);
        let dense2_out = self.dense_2.apply_to(dense1_out);
        let scores = self.scores.apply_to(dense2_out);
        let shape: Ix2 = scores.raw_dim();
        debug_assert_eq!(shape[1], 1);
        let scores = scores.into_shape((shape[0],)).unwrap();
        self.scores_prop_dist.apply_to(scores)
    }
}
