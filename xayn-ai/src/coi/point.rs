use std::time::{Duration, SystemTime};

#[cfg(test)]
use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::{coi::CoiId, embedding::utils::Embedding, utils::system_time_now};

#[derive(Clone, Deserialize, Serialize)]
#[cfg_attr(test, derive(Debug))]
pub(crate) struct CoiView {
    count: usize,
    time: Duration,
    last: SystemTime,
}

impl CoiView {
    pub(crate) fn new(viewed: Option<Duration>) -> Self {
        Self {
            count: 1,
            time: viewed.unwrap_or_default(),
            last: system_time_now(),
        }
    }

    pub(crate) fn update(&mut self, viewed: Option<Duration>) {
        self.count += 1;
        if let Some(viewed) = viewed {
            self.time += viewed;
        }
        self.last = system_time_now();
    }

    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            count: self.count + other.count,
            time: self.time + other.time,
            last: self.last.max(other.last),
        }
    }
}

impl Default for CoiView {
    fn default() -> Self {
        Self {
            count: 1,
            time: Duration::default(),
            last: SystemTime::UNIX_EPOCH,
        }
    }
}

#[obake::versioned]
#[obake(version("0.0.0"))]
#[obake(version("0.1.0"))]
#[obake(version("0.2.0"))]
#[obake(version("0.3.0"))]
#[derive(Clone, Deserialize, Serialize)]
#[cfg_attr(test, derive(Debug, Derivative), derivative(PartialEq))]
pub(crate) struct PositiveCoi {
    #[obake(cfg(">=0.0"))]
    pub(crate) id: CoiId,
    #[obake(cfg(">=0.0"))]
    pub(crate) point: Embedding,
    #[obake(cfg(">=0.3"))]
    #[cfg_attr(test, derivative(PartialEq = "ignore"))]
    pub(crate) view: CoiView,

    // removed fields go below this line
    #[obake(cfg(">=0.0, <0.2"))]
    pub(crate) alpha: f32,
    #[obake(cfg(">=0.0, <0.2"))]
    pub(crate) beta: f32,
}

impl From<PositiveCoi_v0_0_0> for PositiveCoi_v0_1_0 {
    fn from(coi: PositiveCoi_v0_0_0) -> Self {
        Self {
            id: coi.id,
            point: coi.point,
            alpha: coi.alpha,
            beta: coi.beta,
        }
    }
}

impl From<PositiveCoi_v0_1_0> for PositiveCoi_v0_2_0 {
    fn from(coi: PositiveCoi_v0_1_0) -> Self {
        Self {
            id: coi.id,
            point: coi.point,
        }
    }
}

impl From<PositiveCoi_v0_2_0> for PositiveCoi {
    fn from(coi: PositiveCoi_v0_2_0) -> Self {
        Self {
            id: coi.id,
            point: coi.point,
            view: CoiView::default(),
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(crate) struct NegativeCoi {
    pub id: CoiId,
    pub point: Embedding,
}

pub(crate) trait CoiPoint {
    fn new(id: CoiId, point: Embedding, viewed: Option<Duration>) -> Self;
    fn id(&self) -> CoiId;
    fn set_id(&mut self, id: CoiId);
    fn point(&self) -> &Embedding;
    fn set_point(&mut self, embedding: Embedding);
    fn view(&self) -> CoiView {
        CoiView::default()
    }
    fn update_view(&mut self, viewed: Option<Duration>) {
        #![allow(unused_variables)]
    }
}

macro_rules! impl_coi_point {
    (
        $(
            $(#[$attribute:meta])*
            $type:ty {
                fn new(
                    $id:ident: CoiId,
                    $point:ident: Embedding,
                    $new_viewed:ident: Option<Duration> $(,)?
                ) -> Self
                    $new_body:block

                $(
                    fn update_view(
                        $this:ident: &mut Self,
                        $update_viewed:ident: Option<Duration> $(,)?
                    )
                        $update_body:block
                )?
            }
        ),+ $(,)?
    ) => {
        $(
            $(#[$attribute])*
            impl CoiPoint for $type {
                fn new($id: CoiId, $point: Embedding, $new_viewed: Option<Duration>) -> Self
                    $new_body


                fn id(&self) -> CoiId {
                    self.id
                }

                fn set_id(&mut self, id: CoiId) {
                    self.id = id;
                }

                fn point(&self) -> &Embedding {
                    &self.point
                }

                fn set_point(&mut self, embedding: Embedding) {
                    self.point = embedding;
                }

                $(
                    fn update_view($this: &mut Self, $update_viewed: Option<Duration>)
                        $update_body
                )?
            }
        )+
    };
}

impl_coi_point! {
    #[cfg(test)]
    PositiveCoi_v0_0_0 {
        fn new(id: CoiId, point: Embedding, _viewed: Option<Duration>) -> Self {
            Self {
                id,
                point,
                alpha: 1.,
                beta: 1.,
            }
        }
    },

    #[cfg(test)]
    PositiveCoi_v0_1_0 {
        fn new(id: CoiId, point: Embedding, _viewed: Option<Duration>) -> Self {
            Self {
                id,
                point,
                alpha: 1.,
                beta: 1.,
            }
        }
    },

    #[cfg(test)]
    PositiveCoi_v0_2_0 {
        fn new(id: CoiId, point: Embedding, _viewed: Option<Duration>) -> Self {
            Self { id, point }
        }
    },

    PositiveCoi {
        fn new(id: CoiId, point: Embedding, viewed: Option<Duration>) -> Self {
            Self {
                id,
                point,
                view: CoiView::new(viewed),
            }
        }

        fn update_view(self: &mut Self, viewed: Option<Duration>) {
            self.view.update(viewed);
        }
    },

    NegativeCoi {
        fn new(id: CoiId, point: Embedding, _viewed: Option<Duration>) -> Self {
            Self { id, point }
        }
    },
}

// generic types can't be versioned, but aliasing and proper naming in the proc macro call works
#[allow(non_camel_case_types)]
type PositiveCois_v0_0_0 = Vec<PositiveCoi_v0_0_0>;
#[allow(non_camel_case_types)]
type PositiveCois_v0_1_0 = Vec<PositiveCoi_v0_1_0>;
#[allow(non_camel_case_types)]
type PositiveCois_v0_2_0 = Vec<PositiveCoi_v0_2_0>;
#[allow(non_camel_case_types)]
type PositiveCois_v0_3_0 = Vec<PositiveCoi>;

#[obake::versioned]
#[obake(version("0.0.0"))]
#[obake(version("0.1.0"))]
#[obake(version("0.2.0"))]
#[obake(version("0.3.0"))]
#[derive(Clone, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(crate) struct UserInterests {
    #[obake(inherit)]
    #[obake(cfg(">=0.0"))]
    pub positive: PositiveCois,
    #[obake(cfg(">=0.0"))]
    pub negative: Vec<NegativeCoi>,
}

impl From<UserInterests_v0_0_0> for UserInterests_v0_1_0 {
    fn from(ui: UserInterests_v0_0_0) -> Self {
        Self {
            positive: ui.positive.into_iter().map(Into::into).collect(),
            negative: ui.negative.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<UserInterests_v0_1_0> for UserInterests_v0_2_0 {
    fn from(ui: UserInterests_v0_1_0) -> Self {
        Self {
            positive: ui.positive.into_iter().map(Into::into).collect(),
            negative: ui.negative.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<UserInterests_v0_2_0> for UserInterests {
    fn from(ui: UserInterests_v0_2_0) -> Self {
        Self {
            positive: ui.positive.into_iter().map(Into::into).collect(),
            negative: ui.negative.into_iter().map(Into::into).collect(),
        }
    }
}
