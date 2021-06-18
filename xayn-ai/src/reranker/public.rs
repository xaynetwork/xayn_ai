use std::{
    io::{BufRead, Read},
    path::Path,
};

use rubert::{AveragePooler, QAMBert, QAMBertBuilder, SMBert, SMBertBuilder};

use crate::{
    analytics::{Analytics, AnalyticsSystem as AnalyticsSystemImpl},
    coi::{CoiSystem as CoiSystemImpl, Configuration as CoiSystemConfiguration},
    context::Context,
    data::document::{Document, DocumentHistory, RerankingOutcomes},
    ltr::{DomainReranker, LtrBuilder},
    mab::{BetaSampler, MabRanking},
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
    RerankMode,
};

pub struct Systems {
    database: Db,
    smbert: SMBert,
    qambert: QAMBert,
    coi: CoiSystemImpl,
    ltr: DomainReranker,
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

    pub fn rerank(
        &mut self,
        mode: RerankMode,
        history: &[DocumentHistory],
        documents: &[Document],
    ) -> RerankingOutcomes {
        self.0.rerank(mode, history, documents)
    }

    pub fn serialize(&self) -> Result<Vec<u8>, Error> {
        self.0.serialize()
    }
}

pub struct Builder<SV, SM, QAV, QAM, LM> {
    database: Db,
    smbert: SMBertBuilder<SV, SM>,
    qambert: QAMBertBuilder<QAV, QAM>,
    ltr: LtrBuilder<LM>,
}

impl Default for Builder<(), (), (), (), ()> {
    fn default() -> Self {
        Self {
            database: Db::default(),
            smbert: SMBertBuilder::new((), ()),
            qambert: QAMBertBuilder::new((), ()),
            ltr: LtrBuilder::new(()),
        }
    }
}

impl<SV, SM, QAV, QAM, LM> Builder<SV, SM, QAV, QAM, LM> {
    /// Sets the serialized database to use.
    ///
    /// This accepts an option as this makes the builder pattern easier to use
    /// as all consumers (we currently have) want to only optionally set serialized
    /// database.
    pub fn with_serialized_database(
        mut self,
        bytes: Option<impl AsRef<[u8]>>,
    ) -> Result<Self, Error> {
        if let Some(bytes) = bytes {
            self.database = Db::deserialize(bytes.as_ref())?;
        }
        Ok(self)
    }

    pub fn with_smbert_from_reader<W, N>(self, vocab: W, model: N) -> Builder<W, N, QAV, QAM, LM> {
        Builder {
            database: self.database,
            smbert: SMBertBuilder::new(vocab, model),
            qambert: self.qambert,
            ltr: self.ltr,
        }
    }

    pub fn with_smbert_from_file(
        self,
        vocab: impl AsRef<Path>,
        model: impl AsRef<Path>,
    ) -> Result<Builder<impl BufRead, impl Read, QAV, QAM, LM>, Error> {
        Ok(Builder {
            database: self.database,
            smbert: SMBertBuilder::from_files(vocab, model)?,
            qambert: self.qambert,
            ltr: self.ltr,
        })
    }

    pub fn with_qambert_from_reader<W, N>(self, vocab: W, model: N) -> Builder<SV, SM, W, N, LM> {
        Builder {
            database: self.database,
            smbert: self.smbert,
            qambert: QAMBertBuilder::new(vocab, model),
            ltr: self.ltr,
        }
    }

    pub fn with_qambert_from_file(
        self,
        vocab: impl AsRef<Path>,
        model: impl AsRef<Path>,
    ) -> Result<Builder<SV, SM, impl BufRead, impl Read, LM>, Error> {
        Ok(Builder {
            database: self.database,
            smbert: self.smbert,
            qambert: QAMBertBuilder::from_files(vocab, model)?,
            ltr: self.ltr,
        })
    }

    pub fn with_ltr_from_reader<N>(self, model: N) -> Builder<SV, SM, QAV, QAM, N> {
        Builder {
            database: self.database,
            smbert: self.smbert,
            qambert: self.qambert,
            ltr: LtrBuilder::new(model),
        }
    }

    pub fn with_ltr_from_file(
        self,
        model: impl AsRef<Path>,
    ) -> Result<Builder<SV, SM, QAV, QAM, impl Read>, Error> {
        Ok(Builder {
            database: self.database,
            smbert: self.smbert,
            qambert: self.qambert,
            ltr: LtrBuilder::from_file(model)?,
        })
    }

    pub fn build(self) -> Result<Reranker, Error>
    where
        SV: BufRead,
        SM: Read,
        QAV: BufRead,
        QAM: Read,
        LM: Read,
    {
        let database = self.database;
        let smbert = self
            .smbert
            .with_token_size(90)?
            .with_accents(false)
            .with_lowercase(true)
            .with_pooling(AveragePooler)
            .build()?;
        let qambert = self
            .qambert
            .with_token_size(90)?
            .with_accents(false)
            .with_lowercase(true)
            .with_pooling(AveragePooler)
            .build()?;
        let coi = CoiSystemImpl::new(CoiSystemConfiguration::default());
        let ltr = self.ltr.build()?;
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
