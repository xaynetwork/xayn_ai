//! temporary dummy impls, most of this should move to xayn-ai itself

use rubert::RuBert;
use xayn_ai::{
    Analytics,
    AnalyticsSystem,
    BertSystem,
    CoiSystem,
    CoiSystems,
    CommonSystems,
    ConstLtr,
    Context,
    ContextSystem,
    Database,
    DocumentDataWithMab,
    DocumentHistory,
    Error,
    LtrSystem,
    MabRanking,
    MabSystem,
    RerankerData,
};

pub struct DummyDatabase;

impl Database for DummyDatabase {
    fn save_data(&self, state: &RerankerData) -> Result<(), Error> {
        Ok(())
    }

    fn load_data(&self) -> Result<Option<RerankerData>, Error> {
        Ok(None)
    }

    fn save_analytics(&self, _analytics: &Analytics) -> Result<(), Error> {
        Ok(())
    }
}

pub struct DummyAnalytics;

impl AnalyticsSystem for DummyAnalytics {
    fn compute_analytics(
        &self,
        _history: &[DocumentHistory],
        _documents: &[DocumentDataWithMab],
    ) -> Result<Analytics, Error> {
        Ok(Analytics {})
    }
}

pub struct Systems {
    pub database: DummyDatabase,
    pub bert: RuBert,
    pub coi: CoiSystem,
    pub ltr: ConstLtr,
    pub context: Context,
    pub mab: MabRanking,
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
