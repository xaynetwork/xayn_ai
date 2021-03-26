use std::{
    io::{BufRead, Read},
    path::Path,
};

use rubert::{AveragePooler, Builder as RuBertBuilder, RuBert};

use crate::{
    analytics::AnalyticsSystem as AnalyticsSystemImpl,
    coi::{CoiSystem as CoiSystemImpl, Configuration as CoiSystemConfiguration},
    context::Context,
    database::Database,
    ltr::ConstLtr,
    mab::{BetaSampler, MabRanking},
    reranker::Reranker,
    reranker_systems::{
        AnalyticsSystem,
        BertSystem,
        CoiSystem,
        CommonSystems,
        ContextSystem,
        LtrSystem,
        MabSystem,
    },
    Error,
};

pub struct Systems<DB> {
    database: DB,
    bert: RuBert<AveragePooler>,
    coi: CoiSystemImpl,
    ltr: ConstLtr,
    context: Context,
    mab: MabRanking,
    analytics: AnalyticsSystemImpl,
}

impl<DB> CommonSystems for Systems<DB>
where
    DB: Database,
{
    fn database(&self) -> &dyn Database {
        &self.database
    }

    fn bert(&self) -> &dyn BertSystem {
        &self.bert
    }

    fn coi(&self) -> &dyn CoiSystem {
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

pub struct RerankerBuilder {}

impl RerankerBuilder {
    pub fn with_database<DB: Database>(database: DB) -> RerankerBuilderAddBert<DB> {
        RerankerBuilderAddBert { database }
    }
}

pub struct RerankerBuilderAddBert<DB> {
    database: DB,
}

impl<DB> RerankerBuilderAddBert<DB> {
    pub fn with_bert_from_file(
        self,
        vocab: impl AsRef<Path>,
        model: impl AsRef<Path>,
    ) -> Result<RerankerBuilderFinalize<DB>, Error> {
        let bert = make_bert_system(RuBertBuilder::from_files(vocab, model)?)?;

        Ok(RerankerBuilderFinalize {
            systems: make_systems(self.database, bert),
        })
    }

    pub fn with_bert_from_reader<V, M>(
        self,
        vocab: impl BufRead,
        model: impl Read,
    ) -> Result<RerankerBuilderFinalize<DB>, Error> {
        let bert = make_bert_system(RuBertBuilder::new(vocab, model))?;

        Ok(RerankerBuilderFinalize {
            systems: make_systems(self.database, bert),
        })
    }
}

pub struct RerankerBuilderFinalize<DB> {
    systems: Systems<DB>,
}

impl<DB: Database> RerankerBuilderFinalize<DB> {
    pub fn build(self) -> Result<Reranker<Systems<DB>>, Error> {
        Reranker::new(self.systems)
    }

    /// Allows to change the beta sampler of the mab for testing purpose
    #[cfg(test)]
    pub fn with_mab_beta_sampler(mut self, beta_sampler: BetaSampler) -> Self {
        self.systems.mab = MabRanking::new(beta_sampler);

        self
    }
}

fn make_bert_system<V, M, P>(
    builder: RuBertBuilder<V, M, P>,
) -> Result<RuBert<AveragePooler>, Error>
where
    V: BufRead,
    M: Read,
{
    builder
        .with_token_size(90)?
        .with_accents(false)
        .with_lowercase(true)
        .with_pooling(AveragePooler)
        .build()
        .map_err(|e| e.into())
}

fn make_systems<DB>(database: DB, bert: RuBert<AveragePooler>) -> Systems<DB> {
    let ltr = ConstLtr::new();
    let context = Context;
    let mab = MabRanking::new(BetaSampler);
    let coi = CoiSystemImpl::new(CoiSystemConfiguration::default());
    let analytics = AnalyticsSystemImpl;

    Systems {
        database,
        bert,
        coi,
        ltr,
        context,
        mab,
        analytics,
    }
}
