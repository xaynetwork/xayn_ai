use std::{
    io::{BufRead, Read},
    path::Path,
};

use rubert::{AveragePooler, SMBert, SMBertBuilder};

use crate::{
    analytics::{Analytics, AnalyticsSystem as AnalyticsSystemImpl},
    coi::{CoiSystem as CoiSystemImpl, Configuration as CoiSystemConfiguration},
    context::Context,
    data::document::{Document, DocumentHistory, Ranks},
    ltr::ConstLtr,
    mab::{BetaSampler, MabRanking},
    qambert::DummyQAMBert,
    Error,
};

use super::{
    database::{Database, Db},
    systems::{
        AnalyticsSystem,
        CoiSystem,
        CommonSystems,
        ContextSystem,
        LtrSystem,
        MabSystem,
        QAMBertSystem,
        SMBertSystem,
    },
};

pub struct Systems {
    database: Db,
    smbert: SMBert,
    qambert: DummyQAMBert,
    coi: CoiSystemImpl,
    ltr: ConstLtr,
    context: Context,
    mab: MabRanking<BetaSampler>,
    analytics: AnalyticsSystemImpl,
}

impl CommonSystems for Systems {
    fn database(&self) -> &dyn Database {
        &self.database
    }

    fn smbert(&self) -> &dyn SMBertSystem {
        &self.smbert
    }

    fn qambert(&self) -> &dyn QAMBertSystem {
        &self.qambert
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

pub struct Reranker(super::Reranker<Systems>);

impl Reranker {
    pub fn errors(&self) -> &[Error] {
        self.0.errors()
    }

    pub fn analytics(&self) -> Option<&Analytics> {
        self.0.analytics()
    }

    pub fn rerank(&mut self, history: &[DocumentHistory], documents: &[Document]) -> Ranks {
        self.0.rerank(history, documents)
    }

    pub fn serialize(&self) -> Result<Vec<u8>, Error> {
        self.0.serialize()
    }
}

pub struct Builder<V, M> {
    database: Db,
    smbert: SMBertBuilder<V, M>,
}

impl Default for Builder<(), ()> {
    fn default() -> Self {
        Self {
            database: Db::default(),
            smbert: SMBertBuilder::new((), ()),
        }
    }
}

impl<V, M> Builder<V, M> {
    pub fn with_serialized_database(mut self, bytes: &[u8]) -> Result<Self, Error> {
        self.database = Db::deserialize(bytes)?;
        Ok(self)
    }

    pub fn with_smbert_from_reader<W, N>(self, vocab: W, model: N) -> Builder<W, N> {
        Builder {
            database: self.database,
            smbert: SMBertBuilder::new(vocab, model),
        }
    }

    pub fn with_smbert_from_file(
        self,
        vocab: impl AsRef<Path>,
        model: impl AsRef<Path>,
    ) -> Result<Builder<impl BufRead, impl Read>, Error> {
        Ok(Builder {
            database: self.database,
            smbert: SMBertBuilder::from_files(vocab, model)?,
        })
    }

    pub fn build(self) -> Result<Reranker, Error>
    where
        V: BufRead,
        M: Read,
    {
        let database = self.database;
        let smbert = self
            .smbert
            .with_token_size(90)?
            .with_accents(false)
            .with_lowercase(true)
            .with_pooling(AveragePooler)
            .build()?;
        let qambert = DummyQAMBert;
        let coi = CoiSystemImpl::new(CoiSystemConfiguration::default());
        let ltr = ConstLtr::new();
        let context = Context;
        let mab = MabRanking::new(BetaSampler);
        let analytics = AnalyticsSystemImpl;

        super::Reranker::new(Systems {
            database,
            smbert,
            qambert,
            coi,
            ltr,
            context,
            mab,
            analytics,
        })
        .map(Reranker)
    }
}
