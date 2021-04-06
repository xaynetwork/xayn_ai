use std::{
    io::{BufRead, Read},
    path::Path,
};

use rubert::{AveragePooler, Builder as RuBertBuilder, NonePooler, RuBert};

use crate::{
    analytics::AnalyticsSystem as AnalyticsSystemImpl,
    coi::{CoiSystem as CoiSystemImpl, Configuration as CoiSystemConfiguration},
    context::Context,
    database::{DatabaseRaw, Db, Database},
    ltr::ConstLtr,
    mab::{BetaSample, BetaSampler, MabRanking},
    reranker::Reranker,
    reranker_systems::{
        AnalyticsSystem, BertSystem, CoiSystem, CommonSystems, ContextSystem, LtrSystem, MabSystem,
    },
    Error,
};

pub struct Systems<BS, DBR> {
    database: Db<DBR>,
    bert: RuBert<AveragePooler>,
    coi: CoiSystemImpl,
    ltr: ConstLtr,
    context: Context,
    mab: MabRanking<BS>,
    analytics: AnalyticsSystemImpl,
}

impl<BS, DBR> CommonSystems for Systems<BS, DBR>
where
    BS: BetaSample,
    DBR: DatabaseRaw,
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

pub struct Builder<DB, V, M, BS> {
    database: DB,
    bert: RuBertBuilder<V, M, NonePooler>,
    sampler: BS,
}

impl Default for Builder<(), (), (), BetaSampler> {
    fn default() -> Self {
        Self {
            database: (),
            bert: RuBertBuilder::new((), ()),
            sampler: BetaSampler,
        }
    }
}

impl<DBR, V, M, BS> Builder<DBR, V, M, BS> {
    pub fn with_database_raw<D>(self, database: D) -> Builder<D, V, M, BS> {
        Builder {
            database,
            bert: self.bert,
            sampler: self.sampler,
        }
    }

    pub fn with_bert_from_reader<W, N>(self, vocab: W, model: N) -> Builder<DBR, W, N, BS> {
        Builder {
            database: self.database,
            bert: RuBertBuilder::new(vocab, model),
            sampler: self.sampler,
        }
    }

    pub fn with_bert_from_file(
        self,
        vocab: impl AsRef<Path>,
        model: impl AsRef<Path>,
    ) -> Result<Builder<DBR, impl BufRead, impl Read, BS>, Error> {
        Ok(Builder {
            database: self.database,
            bert: RuBertBuilder::from_files(vocab, model)?,
            sampler: self.sampler,
        })
    }

    /// Allows to change the beta sampler of the mab for testing purpose
    #[cfg(test)]
    pub fn with_mab_beta_sampler<NBS>(self, beta_sampler: NBS) -> Builder<DBR, V, M, NBS> {
        Builder {
            database: self.database,
            bert: self.bert,
            sampler: beta_sampler,
        }
    }

    pub fn build(self) -> Result<Reranker<Systems<BS, DBR>>, Error>
    where
        DBR: DatabaseRaw,
        V: BufRead,
        M: Read,
        BS: BetaSample,
    {
        let database = Db::new(self.database);
        let bert = self
            .bert
            .with_token_size(90)?
            .with_accents(false)
            .with_lowercase(true)
            .with_pooling(AveragePooler)
            .build()?;
        let coi = CoiSystemImpl::new(CoiSystemConfiguration::default());
        let ltr = ConstLtr::new();
        let context = Context;
        let mab = MabRanking::new(self.sampler);
        let analytics = AnalyticsSystemImpl;

        Reranker::new(
            Systems {
                database,
                bert,
                coi,
                ltr,
                context,
                mab,
                analytics,
            },
        )
    }
}
