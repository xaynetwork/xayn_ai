pub mod document;
pub(crate) mod document_data;

use serde::{Deserialize, Serialize};

use crate::coi::point::{NegativeCoi, PositiveCois_v0_0_0, PositiveCois_v0_0_1};

#[obake::versioned]
#[obake(version("0.0.0"))]
#[obake(version("0.0.1"))]
#[derive(Clone, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(crate) struct UserInterests {
    #[obake(inherit)]
    #[obake(cfg(">=0.0.0"))]
    pub positive: PositiveCois,
    #[obake(cfg(">=0.0.0"))]
    pub negative: Vec<NegativeCoi>,
}

impl From<UserInterests_v0_0_0> for UserInterests {
    fn from(ui: UserInterests_v0_0_0) -> Self {
        Self {
            positive: ui.positive.into_iter().map(Into::into).collect(),
            negative: ui.negative.into_iter().map(Into::into).collect(),
        }
    }
}
