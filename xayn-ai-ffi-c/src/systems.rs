//! temporary dummy impls, most of this will be moved to xayn-ai itself

use rubert::{AveragePooler, RuBert};
use xayn_ai::{
    Analytics,
    AnalyticsSystem,
    BertSystem,
    BetaSampler,
    CoiSystem,
    CoiSystems,
    CommonSystems,
    ConstLtr,
    Context,
    ContextSystem,
    Database,
    DummyAnalytics,
    Error,
    LtrSystem,
    MabRanking,
    MabSystem,
    RerankerData,
};

pub struct DummyDatabase;

impl Database for DummyDatabase {
    fn save_data(&self, _state: &RerankerData) -> Result<(), Error> {
        Ok(())
    }

    fn load_data(&self) -> Result<Option<RerankerData>, Error> {
        Ok(None)
    }

    fn save_analytics(&self, _analytics: &Analytics) -> Result<(), Error> {
        Ok(())
    }
}

pub struct Systems {
    pub database: DummyDatabase,
    pub bert: RuBert<AveragePooler>,
    pub coi: CoiSystem,
    pub ltr: ConstLtr,
    pub context: Context,
    pub mab: MabRanking<BetaSampler>,
    pub analytics: DummyAnalytics,
}

impl CommonSystems for Systems {
    fn database(&self) -> &dyn Database {
        &self.database
    }

    fn bert(&self) -> &dyn BertSystem {
        &self.bert
    }

    fn coi(&self) -> &dyn CoiSystems {
        &self.coi
    }

    fn ltr(&self) -> &dyn LtrSystem {
        &self.ltr
    }

    fn context(&self) -> &dyn ContextSystem {
        &self.context
    }

    fn mab(&self) -> &dyn MabSystem {
        &self.mab
    }

    fn analytics(&self) -> &dyn AnalyticsSystem {
        &self.analytics
    }
}
