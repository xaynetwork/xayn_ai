use std::{
    io::{BufRead, Read},
    path::Path,
    sync::Arc,
};

use rubert::{AveragePooler, QAMBert, QAMBertBuilder, SMBertBuilder};

use crate::{
    analytics::{Analytics, AnalyticsSystem as AnalyticsSystemImpl},
    coi::{CoiSystem as CoiSystemImpl, Configuration as CoiSystemConfiguration},
    context::Context,
    data::document::{Document, DocumentHistory, RerankingOutcomes},
    embedding::smbert::SMBert,
    error::Error,
    ltr::{DomainReranker, DomainRerankerBuilder},
    reranker::{
        database::{Database, Db},
        systems::{
            AnalyticsSystem,
            CoiSystem,
            CommonSystems,
            ContextSystem,
            LtrSystem,
            QAMBertSystem,
            SMBertSystem,
        },
        RerankMode,
    },
};

pub struct Systems {
    database: Db,
    smbert: SMBert,
    qambert: QAMBert,
    coi: CoiSystemImpl,
    domain: DomainReranker,
    context: Context,
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

    fn mut_coi(&mut self) -> &mut dyn CoiSystem {
        &mut self.coi
    }

    fn ltr(&self) -> &dyn LtrSystem {
        &self.domain
    }

    fn context(&self) -> &dyn ContextSystem {
        &self.context
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

    pub fn syncdata_bytes(&self) -> Result<Vec<u8>, Error> {
        self.0.syncdata_bytes()
    }

    pub fn synchronize(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.0.synchronize(bytes)
    }
}

pub struct Builder<SV, SM, QAV, QAM, DM> {
    database: Db,
    smbert: SMBertBuilder<SV, SM>,
    qambert: QAMBertBuilder<QAV, QAM>,
    domain: DomainRerankerBuilder<DM>,
}

impl Default for Builder<(), (), (), (), ()> {
    fn default() -> Self {
        Self {
            database: Db::default(),
            smbert: SMBertBuilder::new((), ()),
            qambert: QAMBertBuilder::new((), ()),
            domain: DomainRerankerBuilder::new(()),
        }
    }
}

impl<SV, SM, QAV, QAM, DM> Builder<SV, SM, QAV, QAM, DM> {
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

    pub fn with_smbert_from_reader<W, N>(self, vocab: W, model: N) -> Builder<W, N, QAV, QAM, DM> {
        Builder {
            database: self.database,
            smbert: SMBertBuilder::new(vocab, model),
            qambert: self.qambert,
            domain: self.domain,
        }
    }

    pub fn with_smbert_from_file(
        self,
        vocab: impl AsRef<Path>,
        model: impl AsRef<Path>,
    ) -> Result<Builder<impl BufRead, impl Read, QAV, QAM, DM>, Error> {
        Ok(Builder {
            database: self.database,
            smbert: SMBertBuilder::from_files(vocab, model)?,
            qambert: self.qambert,
            domain: self.domain,
        })
    }

    pub fn with_qambert_from_reader<W, N>(self, vocab: W, model: N) -> Builder<SV, SM, W, N, DM> {
        Builder {
            database: self.database,
            smbert: self.smbert,
            qambert: QAMBertBuilder::new(vocab, model),
            domain: self.domain,
        }
    }

    pub fn with_qambert_from_file(
        self,
        vocab: impl AsRef<Path>,
        model: impl AsRef<Path>,
    ) -> Result<Builder<SV, SM, impl BufRead, impl Read, DM>, Error> {
        Ok(Builder {
            database: self.database,
            smbert: self.smbert,
            qambert: QAMBertBuilder::from_files(vocab, model)?,
            domain: self.domain,
        })
    }

    pub fn with_domain_from_reader<N>(self, model: N) -> Builder<SV, SM, QAV, QAM, N> {
        Builder {
            database: self.database,
            smbert: self.smbert,
            qambert: self.qambert,
            domain: DomainRerankerBuilder::new(model),
        }
    }

    pub fn with_domain_from_file(
        self,
        model: impl AsRef<Path>,
    ) -> Result<Builder<SV, SM, QAV, QAM, impl Read>, Error> {
        Ok(Builder {
            database: self.database,
            smbert: self.smbert,
            qambert: self.qambert,
            domain: DomainRerankerBuilder::from_file(model)?,
        })
    }

    pub fn build(self) -> Result<Reranker, Error>
    where
        SV: BufRead,
        SM: Read,
        QAV: BufRead,
        QAM: Read,
        DM: Read,
    {
        let database = self.database;
        let smbert = SMBert::from(Arc::new(
            self.smbert
                .with_token_size(52)?
                .with_accents(false)
                .with_lowercase(true)
                .with_pooling(AveragePooler)
                .build()?,
        ));
        let qambert = self
            .qambert
            .with_token_size(90)?
            .with_accents(false)
            .with_lowercase(true)
            .with_pooling(AveragePooler)
            .build()?;
        let coi = CoiSystemImpl::new(CoiSystemConfiguration::default(), smbert.clone());
        let domain = self.domain.build()?;
        let context = Context;
        let analytics = AnalyticsSystemImpl;

        super::Reranker::new(Systems {
            database,
            smbert,
            qambert,
            coi,
            domain,
            context,
            analytics,
        })
        .map(Reranker)
    }
}
