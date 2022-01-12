use ndarray::arr1;

use crate::{
    analytics::AnalyticsSystem as AnalyticsSys,
    coi::{compute_coi, update_user_interests, Configuration as CoiConfig},
    context::Context,
    data::document_data::{
        DocumentDataWithQAMBert,
        DocumentDataWithSMBert,
        QAMBertComponent,
        SMBertComponent,
    },
    ltr::ConstLtr,
    reranker::{
        database::Database,
        systems::{
            AnalyticsSystem,
            CoiSystem,
            CommonSystems,
            ContextSystem,
            LtrSystem,
            QAMBertSystem,
            SMBertSystem,
        },
    },
    tests::{MemDb, MockCoiSystem, MockQAMBertSystem, MockSMBertSystem},
};

pub(crate) fn mocked_smbert_system() -> MockSMBertSystem {
    let mut mock_smbert = MockSMBertSystem::new();
    mock_smbert.expect_compute_embedding().returning(|docs| {
        Ok(docs
            .iter()
            .map(|doc| {
                let mut embedding: Vec<f32> = doc
                    .document_content
                    .clone()
                    .title
                    .into_bytes()
                    .into_iter()
                    .map(|c| c as f32)
                    .collect();
                embedding.resize(128, 0.);

                DocumentDataWithSMBert {
                    document_base: doc.document_base.clone(),
                    document_content: doc.document_content.clone(),
                    smbert: SMBertComponent {
                        embedding: arr1(&embedding).into(),
                    },
                }
            })
            .collect())
    });
    mock_smbert
}

pub(crate) fn mocked_qambert_system() -> MockQAMBertSystem {
    let mut mock_qambert = MockQAMBertSystem::new();
    mock_qambert.expect_compute_similarity().returning(|docs| {
        Ok(docs
            .iter()
            .map(|doc| {
                let snippet = &doc.document_content.snippet;
                let data = if snippet.is_empty() {
                    &doc.document_content.title
                } else {
                    snippet
                };
                let similarity =
                    (data.len() as f32 - doc.document_content.query_words.len() as f32).abs();

                DocumentDataWithQAMBert {
                    document_base: doc.document_base.clone(),
                    document_content: doc.document_content.clone(),
                    smbert: doc.smbert.clone(),
                    coi: doc.coi.clone(),
                    qambert: QAMBertComponent { similarity },
                }
            })
            .collect())
    });

    mock_qambert
}

fn mocked_coi_system() -> MockCoiSystem {
    let config = CoiConfig::default();
    let neighbors = config.neighbors.get();

    let mut system = MockCoiSystem::new();
    system
        .expect_compute_coi()
        .returning(move |documents, user_interests| {
            compute_coi(documents, user_interests, neighbors)
        });
    system
        .expect_update_user_interests()
        .returning(move |history, documents, user_interests| {
            update_user_interests(
                history,
                documents,
                user_interests,
                |_| unreachable!(),
                config,
            )
        });
    system
}

pub(crate) struct MockCommonSystems<Db, SMBert, QAMBert, Coi, Ltr, Context, Analytics>
where
    Db: Database,
    SMBert: SMBertSystem,
    QAMBert: QAMBertSystem,
    Coi: CoiSystem,
    Ltr: LtrSystem,
    Context: ContextSystem,
    Analytics: AnalyticsSystem,
{
    database: Db,
    smbert: SMBert,
    qambert: QAMBert,
    coi: Coi,
    ltr: Ltr,
    context: Context,
    analytics: Analytics,
}

impl<Db, SMBert, QAMBert, Coi, Ltr, Context, Analytics> CommonSystems
    for MockCommonSystems<Db, SMBert, QAMBert, Coi, Ltr, Context, Analytics>
where
    Db: Database,
    SMBert: SMBertSystem,
    QAMBert: QAMBertSystem,
    Coi: CoiSystem,
    Ltr: LtrSystem,
    Context: ContextSystem,
    Analytics: AnalyticsSystem,
{
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

    fn analytics(&self) -> &dyn AnalyticsSystem {
        &self.analytics
    }
}

impl
    MockCommonSystems<
        MemDb,
        MockSMBertSystem,
        MockQAMBertSystem,
        MockCoiSystem,
        ConstLtr,
        Context,
        AnalyticsSys,
    >
{
    pub(crate) fn new() -> Self {
        Self {
            database: MemDb::new(),
            smbert: mocked_smbert_system(),
            qambert: mocked_qambert_system(),
            coi: mocked_coi_system(),
            ltr: ConstLtr,
            context: Context,
            analytics: AnalyticsSys,
        }
    }
}

impl<Db, SMBert, QAMBert, Coi, Ltr, Context, Analytics>
    MockCommonSystems<Db, SMBert, QAMBert, Coi, Ltr, Context, Analytics>
where
    Db: Database,
    SMBert: SMBertSystem,
    QAMBert: QAMBertSystem,
    Coi: CoiSystem,
    Ltr: LtrSystem,
    Context: ContextSystem,
    Analytics: AnalyticsSystem,
{
    pub(crate) fn set_db<D: Database>(
        self,
        f: impl FnOnce() -> D,
    ) -> MockCommonSystems<D, SMBert, QAMBert, Coi, Ltr, Context, Analytics> {
        MockCommonSystems {
            database: f(),
            smbert: self.smbert,
            qambert: self.qambert,
            coi: self.coi,
            ltr: self.ltr,
            context: self.context,
            analytics: self.analytics,
        }
    }

    pub(crate) fn set_smbert<B: SMBertSystem>(
        self,
        f: impl FnOnce() -> B,
    ) -> MockCommonSystems<Db, B, QAMBert, Coi, Ltr, Context, Analytics> {
        MockCommonSystems {
            database: self.database,
            smbert: f(),
            qambert: self.qambert,
            coi: self.coi,
            ltr: self.ltr,
            context: self.context,
            analytics: self.analytics,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn set_qambert<B: QAMBertSystem>(
        self,
        f: impl FnOnce() -> B,
    ) -> MockCommonSystems<Db, SMBert, B, Coi, Ltr, Context, Analytics> {
        MockCommonSystems {
            database: self.database,
            smbert: self.smbert,
            qambert: f(),
            coi: self.coi,
            ltr: self.ltr,
            context: self.context,
            analytics: self.analytics,
        }
    }

    pub(crate) fn set_coi<C: CoiSystem>(
        self,
        f: impl FnOnce() -> C,
    ) -> MockCommonSystems<Db, SMBert, QAMBert, C, Ltr, Context, Analytics> {
        MockCommonSystems {
            database: self.database,
            smbert: self.smbert,
            qambert: self.qambert,
            coi: f(),
            ltr: self.ltr,
            context: self.context,
            analytics: self.analytics,
        }
    }

    pub(crate) fn set_ltr<L: LtrSystem>(
        self,
        f: impl FnOnce() -> L,
    ) -> MockCommonSystems<Db, SMBert, QAMBert, Coi, L, Context, Analytics> {
        MockCommonSystems {
            database: self.database,
            smbert: self.smbert,
            qambert: self.qambert,
            coi: self.coi,
            ltr: f(),
            context: self.context,
            analytics: self.analytics,
        }
    }

    pub(crate) fn set_context<C: ContextSystem>(
        self,
        f: impl FnOnce() -> C,
    ) -> MockCommonSystems<Db, SMBert, QAMBert, Coi, Ltr, C, Analytics> {
        MockCommonSystems {
            database: self.database,
            smbert: self.smbert,
            qambert: self.qambert,
            coi: self.coi,
            ltr: self.ltr,
            context: f(),
            analytics: self.analytics,
        }
    }

    pub(crate) fn set_analytics<A: AnalyticsSystem>(
        self,
        f: impl FnOnce() -> A,
    ) -> MockCommonSystems<Db, SMBert, QAMBert, Coi, Ltr, Context, A> {
        MockCommonSystems {
            database: self.database,
            smbert: self.smbert,
            qambert: self.qambert,
            coi: self.coi,
            ltr: self.ltr,
            context: self.context,
            analytics: f(),
        }
    }
}

impl Default
    for MockCommonSystems<
        MemDb,
        MockSMBertSystem,
        MockQAMBertSystem,
        MockCoiSystem,
        ConstLtr,
        Context,
        AnalyticsSys,
    >
{
    fn default() -> Self {
        Self::new()
    }
}
