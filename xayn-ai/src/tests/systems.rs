use ndarray::arr1;

use crate::{
    analytics::AnalyticsSystem as AnalyticsSys,
    coi::CoiSystem as CoiSys,
    context::Context,
    data::document_data::{DocumentDataWithSMBert, SMBertEmbeddingComponent},
    ltr::ConstLtr,
    mab::MabRanking,
    reranker::{
        database::Database,
        systems::{
            AnalyticsSystem,
            BertSystem,
            CoiSystem,
            CommonSystems,
            ContextSystem,
            LtrSystem,
            MabSystem,
        },
    },
    tests::{MemDb, MockBertSystem, MockBetaSample},
};

// can later be used for integration tests
// pub fn global_bert_system() -> &'static Arc<RuBert<AveragePooler>> {
//     static BERT_SYSTEM: OnceCell<Arc<RuBert<AveragePooler>>> = OnceCell::new();
//     BERT_SYSTEM.get_or_init(|| {
//         let bert = BertBuilder::from_files(
//             "../data/rubert_v0000/vocab.txt",
//             "../data/rubert_v0000/model.onnx",
//         )
//         .expect("failed to create bert builder from files")
//         .with_token_size(90)
//         .expect("infallible: token size >= 2")
//         .with_accents(false)
//         .with_lowercase(true)
//         .with_pooling(AveragePooler)
//         .build()
//         .expect("failed to build bert");
//         Arc::new(bert)
//     })
// }

// impl BertSystem for Arc<RuBert<AveragePooler>> {
//     fn compute_embedding(
//         &self,
//         documents: Vec<DocumentDataWithDocument>,
//     ) -> Result<Vec<DocumentDataWithEmbedding>, Error> {
//         self.as_ref().compute_embedding(documents)
//     }
// }

pub(crate) fn mocked_bert_system() -> MockBertSystem {
    let mut mock_bert = MockBertSystem::new();
    mock_bert.expect_compute_embedding().returning(|docs| {
        Ok(docs
            .into_iter()
            .map(|doc| {
                let mut embedding: Vec<f32> = doc
                    .document_content
                    .snippet
                    .into_bytes()
                    .into_iter()
                    .map(|c| c as f32)
                    .collect();
                embedding.resize(128, 0.);

                DocumentDataWithSMBert {
                    document_base: doc.document_base,
                    embedding: SMBertEmbeddingComponent {
                        embedding: arr1(&embedding).into(),
                    },
                }
            })
            .collect())
    });
    mock_bert
}

pub(crate) struct MockCommonSystems<Db, Bert, Coi, Ltr, Context, Mab, Analytics>
where
    Db: Database,
    Bert: BertSystem,
    Coi: CoiSystem,
    Ltr: LtrSystem,
    Context: ContextSystem,
    Mab: MabSystem,
    Analytics: AnalyticsSystem,
{
    database: Db,
    bert: Bert,
    coi: Coi,
    ltr: Ltr,
    context: Context,
    mab: Mab,
    analytics: Analytics,
}

impl<Db, Bert, Coi, Ltr, Context, Mab, Analytics> CommonSystems
    for MockCommonSystems<Db, Bert, Coi, Ltr, Context, Mab, Analytics>
where
    Db: Database,
    Bert: BertSystem,
    Coi: CoiSystem,
    Ltr: LtrSystem,
    Context: ContextSystem,
    Mab: MabSystem,
    Analytics: AnalyticsSystem,
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

impl
    MockCommonSystems<
        MemDb,
        MockBertSystem,
        CoiSys,
        ConstLtr,
        Context,
        MabRanking<MockBetaSample>,
        AnalyticsSys,
    >
{
    pub(crate) fn new() -> Self {
        let mut beta_sampler = MockBetaSample::new();
        beta_sampler
            .expect_sample()
            .returning(|alpha, beta| Ok(alpha + beta));

        Self {
            database: MemDb::new(),
            bert: mocked_bert_system(),
            coi: CoiSys::default(),
            ltr: ConstLtr::new(),
            context: Context,
            mab: MabRanking::new(beta_sampler),
            analytics: AnalyticsSys,
        }
    }
}

#[allow(dead_code)]
impl<Db, Bert, Coi, Ltr, Context, Mab, Analytics>
    MockCommonSystems<Db, Bert, Coi, Ltr, Context, Mab, Analytics>
where
    Db: Database,
    Bert: BertSystem,
    Coi: CoiSystem,
    Ltr: LtrSystem,
    Context: ContextSystem,
    Mab: MabSystem,
    Analytics: AnalyticsSystem,
{
    pub(crate) fn set_db<D: Database>(
        self,
        f: impl FnOnce() -> D,
    ) -> MockCommonSystems<D, Bert, Coi, Ltr, Context, Mab, Analytics> {
        MockCommonSystems {
            database: f(),
            bert: self.bert,
            coi: self.coi,
            ltr: self.ltr,
            context: self.context,
            mab: self.mab,
            analytics: self.analytics,
        }
    }

    pub(crate) fn set_bert<B: BertSystem>(
        self,
        f: impl FnOnce() -> B,
    ) -> MockCommonSystems<Db, B, Coi, Ltr, Context, Mab, Analytics> {
        MockCommonSystems {
            database: self.database,
            bert: f(),
            coi: self.coi,
            ltr: self.ltr,
            context: self.context,
            mab: self.mab,
            analytics: self.analytics,
        }
    }

    pub(crate) fn set_coi<C: CoiSystem>(
        self,
        f: impl FnOnce() -> C,
    ) -> MockCommonSystems<Db, Bert, C, Ltr, Context, Mab, Analytics> {
        MockCommonSystems {
            database: self.database,
            bert: self.bert,
            coi: f(),
            ltr: self.ltr,
            context: self.context,
            mab: self.mab,
            analytics: self.analytics,
        }
    }

    pub(crate) fn set_ltr<L: LtrSystem>(
        self,
        f: impl FnOnce() -> L,
    ) -> MockCommonSystems<Db, Bert, Coi, L, Context, Mab, Analytics> {
        MockCommonSystems {
            database: self.database,
            bert: self.bert,
            coi: self.coi,
            ltr: f(),
            context: self.context,
            mab: self.mab,
            analytics: self.analytics,
        }
    }

    pub(crate) fn set_context<C: ContextSystem>(
        self,
        f: impl FnOnce() -> C,
    ) -> MockCommonSystems<Db, Bert, Coi, Ltr, C, Mab, Analytics> {
        MockCommonSystems {
            database: self.database,
            bert: self.bert,
            coi: self.coi,
            ltr: self.ltr,
            context: f(),
            mab: self.mab,
            analytics: self.analytics,
        }
    }

    pub(crate) fn set_mab<M: MabSystem>(
        self,
        f: impl FnOnce() -> M,
    ) -> MockCommonSystems<Db, Bert, Coi, Ltr, Context, M, Analytics> {
        MockCommonSystems {
            database: self.database,
            bert: self.bert,
            coi: self.coi,
            ltr: self.ltr,
            context: self.context,
            mab: f(),
            analytics: self.analytics,
        }
    }

    pub(crate) fn set_analytics<A: AnalyticsSystem>(
        self,
        f: impl FnOnce() -> A,
    ) -> MockCommonSystems<Db, Bert, Coi, Ltr, Context, Mab, A> {
        MockCommonSystems {
            database: self.database,
            bert: self.bert,
            coi: self.coi,
            ltr: self.ltr,
            context: self.context,
            mab: self.mab,
            analytics: f(),
        }
    }
}

impl Default
    for MockCommonSystems<
        MemDb,
        MockBertSystem,
        CoiSys,
        ConstLtr,
        Context,
        MabRanking<MockBetaSample>,
        AnalyticsSys,
    >
{
    fn default() -> Self {
        Self::new()
    }
}
