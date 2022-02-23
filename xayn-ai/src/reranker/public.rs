use std::{io::Read, path::Path, sync::Arc};

use rubert::{AveragePooler, QAMBert, QAMBertConfig, SMBertConfig};

use crate::{
    analytics::{Analytics, AnalyticsSystem as AnalyticsSystemImpl},
    coi::{config::Config as CoiSystemConfig, CoiSystem as CoiSystemImpl},
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

pub struct Builder<'a, SP, QP, DM> {
    database: Db,
    smbert_config: SMBertConfig<'a, SP>,
    qambert_config: QAMBertConfig<'a, QP>,
    domain: DomainRerankerBuilder<DM>,
}

impl<'a, SP, QP> Builder<'a, SP, QP, ()> {
    pub fn from(
        smbert_config: SMBertConfig<'a, SP>,
        qambert_config: QAMBertConfig<'a, QP>,
    ) -> Self {
        Builder {
            database: Db::default(),
            smbert_config,
            qambert_config,
            domain: DomainRerankerBuilder::new(()),
        }
    }
}

impl<'a, SP, QP, DM> Builder<'a, SP, QP, DM> {
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

    pub fn with_domain_from_reader<N>(self, model: N) -> Builder<'a, SP, QP, N> {
        Builder {
            database: self.database,
            smbert_config: self.smbert_config,
            qambert_config: self.qambert_config,
            domain: DomainRerankerBuilder::new(model),
        }
    }

    pub fn with_domain_from_file(
        self,
        model: impl AsRef<Path>,
    ) -> Result<Builder<'a, SP, QP, impl Read>, Error> {
        Ok(Builder {
            database: self.database,
            smbert_config: self.smbert_config,
            qambert_config: self.qambert_config,
            domain: DomainRerankerBuilder::from_file(model)?,
        })
    }

    pub fn build(self) -> Result<Reranker, Error>
    where
        DM: Read,
    {
        let smbert_config = self
            .smbert_config
            .with_token_size(52)?
            .with_accents(false)
            .with_lowercase(true)
            .with_pooling(AveragePooler);

        let smbert_pipeline = rubert::SMBert::from(smbert_config)?;
        let smbert = SMBert::from(Arc::new(smbert_pipeline));

        let qambert_config = self
            .qambert_config
            .with_token_size(90)?
            .with_accents(false)
            .with_lowercase(true)
            .with_pooling(AveragePooler);

        let coi = CoiSystemImpl::new(CoiSystemConfig::default(), smbert.clone());
        let domain = self.domain.build()?;

        super::Reranker::new(Systems {
            database: self.database,
            smbert,
            qambert: QAMBert::from(qambert_config)?,
            coi,
            domain,
            context: Context,
            analytics: AnalyticsSystemImpl,
        })
        .map(Reranker)
    }
}
