use std::{
    io::{BufRead, Read},
    path::Path,
};

use rubert::{AveragePooler, Builder as RuBertBuilder, NonePooler, RuBert};

use crate::{
    analytics::{Analytics, AnalyticsSystem as AnalyticsSystemImpl},
    coi::{CoiSystem as CoiSystemImpl, Configuration as CoiSystemConfiguration},
    context::Context,
    data::document::{Document, DocumentHistory, DocumentsRank},
    database::{Database, DatabaseRaw, Db},
    ltr::ConstLtr,
    mab::{BetaSampler, MabRanking},
    reranker,
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

pub struct Systems<DBR> {
    database: Db<DBR>,
    bert: RuBert<AveragePooler>,
    coi: CoiSystemImpl,
    ltr: ConstLtr,
    context: Context,
    mab: MabRanking<BetaSampler>,
    analytics: AnalyticsSystemImpl,
}

impl<DBR> CommonSystems for Systems<DBR>
where
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

pub struct Reranker<DBR>(reranker::Reranker<Systems<DBR>>);

impl<DBR> Reranker<DBR>
where
    DBR: DatabaseRaw,
{
    pub fn errors(&self) -> &Vec<Error> {
        self.0.errors()
    }

    pub fn analytics(&self) -> &Option<Analytics> {
        self.0.analytics()
    }

    pub fn rerank(&mut self, history: &[DocumentHistory], documents: &[Document]) -> DocumentsRank {
        self.0.rerank(history, documents)
    }
}

pub struct Builder<DB, V, M> {
    database: DB,
    bert: RuBertBuilder<V, M, NonePooler>,
}

impl Default for Builder<(), (), ()> {
    fn default() -> Self {
        Self {
            database: (),
            bert: RuBertBuilder::new((), ()),
        }
    }
}

impl<DBR, V, M> Builder<DBR, V, M> {
    pub fn with_database_raw<D>(self, database: D) -> Builder<D, V, M> {
        Builder {
            database,
            bert: self.bert,
        }
    }

    pub fn with_bert_from_reader<W, N>(self, vocab: W, model: N) -> Builder<DBR, W, N> {
        Builder {
            database: self.database,
            bert: RuBertBuilder::new(vocab, model),
        }
    }

    pub fn with_bert_from_file(
        self,
        vocab: impl AsRef<Path>,
        model: impl AsRef<Path>,
    ) -> Result<Builder<DBR, impl BufRead, impl Read>, Error> {
        Ok(Builder {
            database: self.database,
            bert: RuBertBuilder::from_files(vocab, model)?,
        })
    }

    pub fn build(self) -> Result<Reranker<DBR>, Error>
    where
        DBR: DatabaseRaw,
        V: BufRead,
        M: Read,
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
        let mab = MabRanking::new(BetaSampler);
        let analytics = AnalyticsSystemImpl;

        reranker::Reranker::new(Systems {
            database,
            bert,
            coi,
            ltr,
            context,
            mab,
            analytics,
        })
        .map(Reranker)
    }
}
