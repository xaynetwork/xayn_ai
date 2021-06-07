//! The AI and its I/O types.

pub(crate) mod ai;
pub(crate) mod analytics;
pub(crate) mod bytes;

#[cfg(doc)]
pub use self::{
    ai::{
        xaynai_analytics,
        xaynai_drop,
        xaynai_faults,
        xaynai_new,
        xaynai_rerank,
        xaynai_serialize,
        CXaynAi,
    },
    analytics::{analytics_drop, CAnalytics},
    bytes::{bytes_drop, bytes_new, CBytes},
};
