//! ListNet implementation using the NdArray crate.

mod data;
mod model;
mod optimizer;
#[cfg(test)]
mod tests;
mod trainer;

pub use self::{
    data::{
        prepare_inputs,
        prepare_target_prob_dist,
        DataSource,
        GradientSet,
        SampleOwned,
        SampleView,
    },
    model::ListNet,
    optimizer::MiniBatchSgd,
    trainer::{ListNetTrainer, TrainingController},
};
