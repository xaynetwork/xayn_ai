// all warning are errors, we can enforce this here instead of -D warning on the command line
#![deny(warnings)]
#![deny(
    // pub items must have documentation
    missing_docs,
    // warn in case like &T.clone() where T: !Send, in this case the reference will be cloned
    // but is not the intended meaning
    noop_method_call,
    // some warning are error in the compiler with 1.56 but there are still warn lints that we
    // need to fix
    rust_2018_idioms,
    // warn about code not compatible with edition 2021
    // rust_2021_compatibility, we can only use it when we switch to 1.55 and remove it after 1.56
    // some warning are error in the compiler with 1.56 but there are still warn lints that we
    // need to fix
    rust_2018_idioms,
    // pub items that are not reachable from outside and thus should be pub(crate)
    // sealed trait can set this to allow
    unreachable_pub,
    unsafe_code,
    unsafe_op_in_unsafe_fn,
    // warn when we import a name but we use it qualified: use foo::bar; fn f() { foo::bar }
    unused_qualifications,
)]
// enable a lot of linters
// most warn that we have at the moment:
// use _ in numbers. 10000 => 10_000
// variable name to similar: doc & dom, coi1 & cois
// struct/enum name starts/end with the name of the module documents.rs => DocumentId
// we can decide to allows some of these if they are too annoying
#![deny(clippy::pedantic)]
// if future is not send it cannot be sent to another thread reducing performance
#![deny(clippy::future_not_send)]

mod analytics;
mod coi;
mod context;
mod data;
mod embedding;
mod error;
mod ltr;
mod reranker;
mod utils;

pub use crate::{
    analytics::Analytics,
    coi::CoiId,
    data::document::{
        DayOfWeek,
        Document,
        DocumentHistory,
        DocumentId,
        QueryId,
        Relevance,
        RerankingOutcomes,
        SessionId,
        UserAction,
        UserFeedback,
    },
    error::Error,
    reranker::{
        public::{Builder, Reranker},
        RerankMode,
    },
};

#[cfg(test)]
mod tests;

// we need to export rstest_reuse from the root for it to work.
// `use rstest_reuse` will trigger `clippy::single_component_path_imports`
// which is not possible to silence.
#[cfg(test)]
#[allow(unused_imports)]
#[rustfmt::skip]
pub(crate) use rstest_reuse as rstest_reuse;

// Reexport for the dev-tool
#[doc(hidden)]
pub use crate::ltr::{list_net, list_net_training_data_from_history};
