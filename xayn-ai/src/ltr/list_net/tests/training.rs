use std::{
    convert::{Infallible, TryInto},
    env,
    ffi::OsString,
    path::PathBuf,
};

use ndarray::{arr1, arr2, Array};

use super::{
    super::{
        ndutils::io::{FlattenedArray, UnexpectedNumberOfDimensions},
        optimizer::MiniBatchSgd,
    },
    inference::{SAMPLE_INPUTS, SAMPLE_INPUTS_TO_FEW},
    *,
};

struct VecDataSource {
    batch_size: usize,
    training_data_idx: usize,
    training_data: Vec<(Array2<f32>, Array1<f32>)>,
    evaluation_data_idx: usize,
    evaluation_data: Vec<(Array2<f32>, Array1<f32>)>,
}

impl VecDataSource {
    fn new(
        training_data: Vec<(Array2<f32>, Array1<f32>)>,
        evaluation_data: Vec<(Array2<f32>, Array1<f32>)>,
    ) -> Self {
        Self {
            batch_size: 0,
            training_data_idx: 0,
            training_data,
            evaluation_data_idx: 0,
            evaluation_data,
        }
    }
}

#[derive(Error, Debug)]
#[error("Batch Size 0 is not supported (or reset was not called)")]
struct BatchSize0Error;

impl DataSource for VecDataSource {
    type Error = BatchSize0Error;

    fn reset(&mut self, batch_size: usize) -> Result<usize, Self::Error> {
        if batch_size == 0 {
            return Err(BatchSize0Error);
        }
        self.batch_size = batch_size;
        self.training_data_idx = 0;
        self.evaluation_data_idx = 0;
        Ok(self.training_data.len() / batch_size)
    }

    fn next_training_batch(&mut self) -> Result<Vec<Sample>, Self::Error> {
        if self.batch_size == 0 {
            return Err(BatchSize0Error);
        }

        let end_idx = self.training_data_idx + self.batch_size;
        if end_idx <= self.training_data.len() {
            let start_idx = self.training_data_idx;
            self.training_data_idx = end_idx;

            let samples = self.training_data[start_idx..end_idx]
                .iter()
                .map(|(inputs, target_prop_dist)| Sample {
                    inputs: inputs.view(),
                    target_prob_dist: target_prop_dist.view(),
                })
                .collect();
            Ok(samples)
        } else {
            Ok(Vec::new())
        }
    }

    fn next_evaluation_sample(&mut self) -> Result<Option<Sample>, Self::Error> {
        if self.evaluation_data_idx < self.evaluation_data.len() {
            let idx = self.evaluation_data_idx;
            self.evaluation_data_idx += 1;

            let data = &self.evaluation_data[idx];
            Ok(Some(Sample {
                inputs: data.0.view(),
                target_prob_dist: data.1.view(),
            }))
        } else {
            Ok(None)
        }
    }
}

struct TestController {
    evaluation_results: Vec<Option<f32>>,
}

impl TestController {
    fn new() -> Self {
        Self {
            evaluation_results: Vec::new(),
        }
    }
}

impl TrainingController for TestController {
    type Error = Infallible;

    type Outcome = (Self, ListNet);

    fn begin_of_batch(&mut self) -> Result<(), Self::Error> {
        eprintln!("begin batch");
        Ok(())
    }

    fn end_of_batch(&mut self, losses: Vec<f32>) -> Result<(), Self::Error> {
        eprintln!("end of batch");
        dbg!(losses);
        Ok(())
    }

    fn begin_of_epoch(
        &mut self,
        _nr_batches: usize,
        _list_net: &ListNet,
    ) -> Result<(), Self::Error> {
        eprintln!("begin of epoch");
        Ok(())
    }

    fn end_of_epoch(
        &mut self,
        _list_net: &ListNet,
        mean_kv_divergence_evaluation: Option<f32>,
    ) -> Result<(), Self::Error> {
        eprintln!("end of epoch");
        dbg!(mean_kv_divergence_evaluation);
        self.evaluation_results.push(mean_kv_divergence_evaluation);
        Ok(())
    }

    fn begin_of_training(
        &mut self,
        _nr_epochs: usize,
        _list_net: &ListNet,
    ) -> Result<(), Self::Error> {
        eprintln!("begin of training");
        Ok(())
    }

    fn end_of_training(&mut self) -> Result<(), Self::Error> {
        eprintln!("end of training");
        Ok(())
    }

    fn training_result(self, list_net: ListNet) -> Result<Self::Outcome, Self::Error> {
        Ok((self, list_net))
    }
}

#[test]
fn test_training_list_net_is_reproducible_for_same_inputs_and_state() {
    use Relevance::{High, Low, Medium};
    let list_net = LIST_NET.clone();

    let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
        .into_shape((10, 50))
        .unwrap();

    let relevances = vec![Low, Low, Medium, Medium, Low, Medium, High, Low, High, High];
    let data_frame = (inputs, prepare_target_prob_dist(&relevances).unwrap());

    let training_data = vec![data_frame.clone(), data_frame.clone(), data_frame.clone()];
    let test_data = vec![data_frame];
    let nr_epochs = 5;
    let batch_size = 1;

    let (ctrl1, ln1) = {
        let data_source = VecDataSource::new(training_data.clone(), test_data.clone());
        let callbacks = TestController::new();
        let optimizer = MiniBatchSgd { learning_rate: 0.1 };
        let trainer = ListNetTrainer::new(list_net.clone(), data_source, callbacks, optimizer);
        trainer.train(nr_epochs, batch_size).unwrap()
    };
    let (ctrl2, ln2) = {
        let data_source = VecDataSource::new(training_data, test_data);
        let callbacks = TestController::new();
        let optimizer = MiniBatchSgd { learning_rate: 0.1 };
        let trainer = ListNetTrainer::new(list_net, data_source, callbacks, optimizer);
        trainer.train(nr_epochs, batch_size).unwrap()
    };

    assert_approx_eq!(f32, ln1.dense1.weights(), ln2.dense1.weights());
    assert_approx_eq!(f32, ln1.dense1.bias(), ln2.dense1.bias());
    assert_approx_eq!(f32, ln1.dense2.weights(), ln2.dense2.weights());
    assert_approx_eq!(f32, ln1.dense2.bias(), ln2.dense2.bias());
    assert_approx_eq!(f32, ln1.scores.weights(), ln2.scores.weights());
    assert_approx_eq!(f32, ln1.scores.bias(), ln2.scores.bias());
    assert_approx_eq!(f32, ln1.prob_dist.weights(), ln2.prob_dist.weights());
    assert_approx_eq!(f32, ln1.prob_dist.bias(), ln2.prob_dist.bias());

    assert_approx_eq!(f32, &ctrl1.evaluation_results, &ctrl2.evaluation_results);

    assert!(
        ctrl1
            .evaluation_results
            .iter()
            .all(|v| !v.unwrap().is_nan()),
        "contains NaN values {:?}",
        ctrl1.evaluation_results
    );
}

struct BinParamsEqTestGuard {
    params: BinParams,
    write_to_path_on_normal_drop: Option<PathBuf>,
    nr_epochs: usize,
    next_epoch: usize,
    epsilon: f32,
    ulps: i32,
}

impl BinParamsEqTestGuard {
    fn setup(path: impl AsRef<Path>, nr_epochs: usize, epsilon: f32, ulps: i32) -> Self {
        let rewrite_instead_of_test =
            env::var_os("LTR_LIST_NET_TRAINING_INTERMEDIATES_REWRITE") == Some(OsString::from("1"));
        if dbg!(rewrite_instead_of_test) {
            Self {
                params: BinParams::default(),
                write_to_path_on_normal_drop: Some(path.as_ref().to_owned()),
                nr_epochs,
                next_epoch: 0,
                epsilon,
                ulps,
            }
        } else {
            Self {
                params: BinParams::deserialize_from_file(path).unwrap(),
                write_to_path_on_normal_drop: None,
                nr_epochs,
                next_epoch: 0,
                epsilon,
                ulps,
            }
        }
    }

    fn next_iteration(&mut self) -> bool {
        if self.next_epoch < self.nr_epochs {
            self.next_epoch += 1;
            dbg!(self.next_epoch);
            true
        } else {
            false
        }
    }

    fn do_rewrite(&self) -> bool {
        self.write_to_path_on_normal_drop.is_some()
    }

    fn assert_array_eq<D>(&mut self, prefix: Option<&str>, name: &str, array: &Array<f32, D>)
    where
        FlattenedArray<f32>:
            TryInto<Array<f32, D>, Error = UnexpectedNumberOfDimensions> + From<Array<f32, D>>,
        D: Dimension,
    {
        let do_rewrite = self.do_rewrite();
        let mut params = self
            .params
            .with_scope(&format!("{}", self.next_epoch.saturating_sub(1)));

        let mut params = if let Some(prefix) = prefix {
            params.with_scope(prefix)
        } else {
            params
        };

        if do_rewrite {
            params.insert(name, array.clone());
        } else {
            dbg!(params.create_name(name));
            let expected = params.take::<Array<f32, D>>(name).unwrap();
            assert_approx_eq!(
                f32,
                array,
                expected,
                epsilon = self.epsilon,
                ulps = self.ulps
            );
        }
    }
}

impl Drop for BinParamsEqTestGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }

        if let Some(path) = self.write_to_path_on_normal_drop.take() {
            self.params.serialize_into_file(path).unwrap();
        }
    }
}

macro_rules! assert_trace_array {
    ($inout:ident =?= $($array:ident),+) => ($(
        $inout.assert_array_eq(None, stringify!($array), &$array);
    )*);

    ($inout:ident [$($idx:expr),+] =?= $($array:expr),+) => ({
        let prefix = [$($idx.to_string()),*].join("/");
        $($inout.assert_array_eq(Some(&prefix), stringify!($array), &$array);)*
    });
}

#[test]
fn test_training_with_preset_initial_state_and_input_produces_expected_results() {
    use Relevance::{High, Low, Medium};

    let mut list_net = LIST_NET.clone();
    let mut reference_list_net = LIST_NET.clone();

    let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
        .into_shape((10, 50))
        .unwrap();

    let relevances = vec![Low, Low, Medium, Medium, Low, Medium, High, Low, High, High];

    let mut test_guard = BinParamsEqTestGuard::setup(
        "../data/ltr_test_data_v0000/check_training_intermediates.binparams",
        4,
        0.0001,
        8,
    );

    while test_guard.next_iteration() {
        // Inlined all functions involved into training to get *all* intermediates.
        let target_prob_dist = prepare_target_prob_dist(&relevances).unwrap();
        assert_trace_array!(test_guard =?= target_prob_dist);
        let (dense1_y, dense1_z) = list_net.dense1.run(&inputs, true);
        let dense1_z = dense1_z.unwrap();
        assert_trace_array!(test_guard =?= dense1_y, dense1_z);
        let (dense2_y, dense2_z) = list_net.dense2.run(&dense1_y, true);
        let dense2_z = dense2_z.unwrap();
        assert_trace_array!(test_guard =?= dense2_y, dense2_z);
        let (scores_y, scores_z) = list_net.scores.run(&dense2_y, true);
        let scores_z = scores_z.unwrap();
        assert_trace_array!(test_guard =?= scores_y, scores_z);

        let scores_y = scores_y.index_axis_move(Axis(1), 0);

        let (prob_dist_y, prob_dist_z) = list_net.prob_dist.run(&scores_y, true);
        let prob_dist_z = prob_dist_z.unwrap();
        assert_trace_array!(test_guard =?= prob_dist_y, prob_dist_z);

        let nr_documents = inputs.shape()[0];
        let p_cost_and_prob_dist = prob_dist_y - &target_prob_dist;

        let d_prob_dist = list_net
            .prob_dist
            .gradients_from_partials_1d(scores_y.view(), p_cost_and_prob_dist.view());

        let p_scores = list_net.prob_dist.weights().dot(&p_cost_and_prob_dist);

        let mut d_scores = Vec::new();
        let mut d_dense2 = Vec::new();
        let mut d_dense1 = Vec::new();

        for row in 0..nr_documents {
            // From here on training is "split" into multiple parallel "path" using
            // shared weights (hence why we add up the gradients).
            let p_scores = p_scores.slice(s![row..row + 1]);
            assert_trace_array!(test_guard [row] =?= p_scores.to_owned());

            let d_scores_part = list_net
                .scores
                .gradients_from_partials_1d(dense2_y.slice(s![row, ..]), p_scores);
            assert_trace_array!(test_guard [row] =?= d_scores_part.weight_gradients, d_scores_part.bias_gradients);
            d_scores.push(d_scores_part);

            let p_dense2 = list_net.scores.weights().dot(&p_scores)
                * Relu::partial_derivatives_at(&dense2_z.slice(s![row, ..]));
            assert_trace_array!(test_guard [row] =?= p_dense2);
            let d_dense2_part = list_net
                .dense2
                .gradients_from_partials_1d(dense1_y.slice(s![row, ..]), p_dense2.view());
            assert_trace_array!(test_guard [row] =?= d_dense2_part.weight_gradients, d_dense2_part.bias_gradients);
            d_dense2.push(d_dense2_part);

            let p_dense1 = list_net.dense2.weights().dot(&p_dense2)
                * Relu::partial_derivatives_at(&dense1_z.slice(s![row, ..]));
            assert_trace_array!(test_guard [row] =?= p_dense1);
            let d_dense1_part = list_net
                .dense1
                .gradients_from_partials_1d(inputs.slice(s![row, ..]), p_dense1.view());
            assert_trace_array!(test_guard [row] =?= d_dense1_part.weight_gradients, d_dense1_part.bias_gradients);
            d_dense1.push(d_dense1_part);

            let d_scores = d_scores.last().unwrap();
            assert_trace_array!(test_guard [row, "gradients"] =?= d_scores.weight_gradients, d_scores.bias_gradients);
            let d_dense2 = d_dense2.last().unwrap();
            assert_trace_array!(test_guard [row, "gradients"] =?= d_dense2.weight_gradients, d_dense2.bias_gradients);
            let d_dense1 = d_dense1.last().unwrap();
            assert_trace_array!(test_guard [row, "gradients"] =?= d_dense1.weight_gradients, d_dense1.bias_gradients);
        }

        let d_scores = DenseGradientSet::merge_shared(d_scores).unwrap();
        let d_dense2 = DenseGradientSet::merge_shared(d_dense2).unwrap();
        let d_dense1 = DenseGradientSet::merge_shared(d_dense1).unwrap();

        assert_trace_array!(test_guard ["final", "gradients"] =?= d_prob_dist.weight_gradients, d_prob_dist.bias_gradients);
        assert_trace_array!(test_guard ["final", "gradients"] =?= d_scores.weight_gradients, d_scores.bias_gradients);
        assert_trace_array!(test_guard ["final", "gradients"] =?= d_dense2.weight_gradients, d_dense2.bias_gradients);
        assert_trace_array!(test_guard ["final", "gradients"] =?= d_dense1.weight_gradients, d_dense1.bias_gradients);

        let mut gradients = GradientSet {
            dense1: d_dense1,
            dense2: d_dense2,
            scores: d_scores,
            prob_dist: d_prob_dist,
        };

        gradients *= -0.1;
        list_net.add_gradients(gradients);

        // Check if our implementation diverged from the inlined and extended code above.
        let (mut gradients, _) = reference_list_net.gradients_for_query(Sample {
            inputs: inputs.view(),
            target_prob_dist: target_prob_dist.view(),
        });
        gradients *= -0.1;
        reference_list_net.add_gradients(gradients);

        assert_approx_eq!(
            f32,
            list_net.prob_dist.weights(),
            reference_list_net.prob_dist.weights()
        );
        assert_approx_eq!(
            f32,
            list_net.prob_dist.bias(),
            reference_list_net.prob_dist.bias()
        );
        assert_approx_eq!(
            f32,
            list_net.scores.weights(),
            reference_list_net.scores.weights()
        );
        assert_approx_eq!(
            f32,
            list_net.scores.bias(),
            reference_list_net.scores.bias()
        );
        assert_approx_eq!(
            f32,
            list_net.dense2.weights(),
            reference_list_net.dense2.weights()
        );
        assert_approx_eq!(
            f32,
            list_net.dense2.bias(),
            reference_list_net.dense2.bias()
        );
        assert_approx_eq!(
            f32,
            list_net.dense1.weights(),
            reference_list_net.dense1.weights()
        );
        assert_approx_eq!(
            f32,
            list_net.dense1.bias(),
            reference_list_net.dense1.bias()
        );
    }
}

#[test]
fn test_training_list_net_does_not_panic() {
    use Relevance::{High, Low, Medium};

    let list_net = LIST_NET.clone();

    let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
        .into_shape((10, 50))
        .unwrap();

    // Not very good checksum :-) (sanity check)
    let sum: f32 = inputs.iter().sum();
    assert_approx_eq!(f32, sum, 1666.1575);

    let relevances = vec![Low, Low, Medium, Medium, Low, Medium, High, Low, High, High];
    let data_frame = (inputs, prepare_target_prob_dist(&relevances).unwrap());

    let training_data = vec![data_frame.clone(), data_frame.clone(), data_frame.clone()];
    let test_data = vec![data_frame];

    let nr_epochs = 5;
    let data_source = VecDataSource::new(training_data, test_data);
    let callbacks = TestController::new();
    let optimizer = MiniBatchSgd { learning_rate: 0.1 };
    let trainer = ListNetTrainer::new(list_net, data_source, callbacks, optimizer);
    let (controller, _list_net) = trainer.train(nr_epochs, 3).unwrap();
    let evaluation_results = controller.evaluation_results;

    assert_eq!(evaluation_results.len(), nr_epochs);
    assert!(
        evaluation_results.iter().all(|v| !v.unwrap().is_nan()),
        "contains NaN values {:?}",
        evaluation_results
    );
    assert!(
        evaluation_results.first() > evaluation_results.last(),
        "unexpected regression of training: {:?}",
        evaluation_results
    );
}

#[test]
fn test_gradients_merge_batch() {
    let res = GradientSet::mean_of(vec![]);
    assert!(res.is_none());

    let a = GradientSet {
        dense1: DenseGradientSet {
            weight_gradients: arr2(&[[0.1, -2.], [0.3, 0.04]]),
            bias_gradients: arr1(&[0.4, 1.23]),
        },
        dense2: DenseGradientSet {
            weight_gradients: arr2(&[[2., -2.], [-0.3, 0.4]]),
            bias_gradients: arr1(&[0.1, 3.43]),
        },
        scores: DenseGradientSet {
            weight_gradients: arr2(&[[0.125, 2.4], [0.321, 0.454]]),
            bias_gradients: arr1(&[0.42, 2.23]),
        },
        prob_dist: DenseGradientSet {
            weight_gradients: arr2(&[[100., -0.2], [3.25, 0.22]]),
            bias_gradients: arr1(&[-0.42, -2.25]),
        },
    };

    let a2 = GradientSet::mean_of(vec![a.clone()]).unwrap();

    assert_approx_eq!(f32, &a2.dense1.weight_gradients, &a.dense1.weight_gradients);
    assert_approx_eq!(f32, &a2.dense1.bias_gradients, &a.dense1.bias_gradients);
    assert_approx_eq!(f32, &a2.dense2.weight_gradients, &a.dense2.weight_gradients);
    assert_approx_eq!(f32, &a2.dense2.bias_gradients, &a.dense2.bias_gradients);
    assert_approx_eq!(f32, &a2.scores.weight_gradients, &a.scores.weight_gradients);
    assert_approx_eq!(f32, &a2.scores.bias_gradients, &a.scores.bias_gradients);
    assert_approx_eq!(
        f32,
        &a2.prob_dist.weight_gradients,
        &a.prob_dist.weight_gradients
    );
    assert_approx_eq!(
        f32,
        &a2.prob_dist.bias_gradients,
        &a.prob_dist.bias_gradients
    );

    let b = GradientSet {
        dense1: DenseGradientSet {
            weight_gradients: arr2(&[[0.1, 2.], [0.3, 0.04]]),
            bias_gradients: arr1(&[0.4, 1.23]),
        },
        dense2: DenseGradientSet {
            weight_gradients: arr2(&[[0.2, -2.8], [0.3, 0.04]]),
            bias_gradients: arr1(&[0.4, 1.23]),
        },
        scores: DenseGradientSet {
            weight_gradients: arr2(&[[0.1, -2.], [0.3, 0.04]]),
            bias_gradients: arr1(&[0.4, 1.23]),
        },
        prob_dist: DenseGradientSet {
            weight_gradients: arr2(&[[0.0, -2.], [0.3, 0.04]]),
            bias_gradients: arr1(&[0.38, 1.21]),
        },
    };

    let g = GradientSet::mean_of(vec![a, b]).unwrap();

    assert_approx_eq!(f32, &g.dense1.weight_gradients, [[0.1, 0.], [0.3, 0.04]]);
    assert_approx_eq!(f32, &g.dense1.bias_gradients, [0.4, 1.23]);
    assert_approx_eq!(f32, &g.dense2.weight_gradients, [[1.1, -2.4], [0.0, 0.22]]);
    assert_approx_eq!(f32, &g.dense2.bias_gradients, [0.25, 2.33]);
    assert_approx_eq!(
        f32,
        &g.scores.weight_gradients,
        [[0.1125, 0.2], [0.3105, 0.247]],
        ulps = 4
    );
    assert_approx_eq!(f32, &g.scores.bias_gradients, [0.41, 1.73]);
    assert_approx_eq!(
        f32,
        &g.prob_dist.weight_gradients,
        [[50., -1.1], [1.775, 0.13]]
    );
    assert_approx_eq!(f32, &g.prob_dist.bias_gradients, [-0.02, -0.52], ulps = 4);
}

#[test]
fn test_prepare_target_prop_dist() {
    use Relevance::{High, Low, Medium};

    let relevances = vec![Low; 10];
    let res = prepare_target_prob_dist(&relevances);
    assert!(res.is_none());

    let relevances = vec![Low, Low, Medium, Medium, Low, Medium, High, Low, High, High];
    let dist = prepare_target_prob_dist(&relevances).unwrap();
    assert_approx_eq!(
        f32,
        dist,
        [
            0.051_708_337,
            0.046_787_64,
            0.115_079_02,
            0.104_127_81,
            0.034_661_137,
            0.085_252_635,
            0.209_687_65,
            0.025_677_6,
            0.171_677_72,
            0.155_340_43,
        ]
    );
}

#[test]
fn test_prepare_inputs() {
    let inputs = Array1::from(SAMPLE_INPUTS.to_vec())
        .into_shape((10, 50))
        .unwrap();

    let processed_inputs = prepare_inputs(&inputs).unwrap();

    assert_approx_eq!(f32, &inputs, processed_inputs);

    let few_inputs = Array1::from(SAMPLE_INPUTS_TO_FEW.to_vec())
        .into_shape((3, 50))
        .unwrap();

    assert!(prepare_inputs(&few_inputs).is_none());
}
