#![allow(dead_code)]

#[repr(transparent)]
#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub struct DocumentId(pub String);

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub(crate) struct DocumentComponent {
    pub id: DocumentId,
    pub snippet: String,
}

pub(crate) trait GetDocumentComponent {
    fn document(&self) -> &DocumentComponent;
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub(crate) struct LtrComponent {
    pub context_value: f32,
}

pub(crate) trait GetLtrComponent {
    fn ltr(&self) -> &LtrComponent;
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub(crate) struct EmbeddingComponent {
    pub embedding: Vec<f32>,
}

pub(crate) trait GetEmbeddingComponent {
    fn embedding(&self) -> &EmbeddingComponent;
}

#[repr(transparent)]
#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub(crate) struct CenterOfInterestId(pub usize);

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub(crate) struct CenterOfInterestComponent {
    pub(crate) id: CenterOfInterestId,
    /// Distance from the positive center of interest
    pub(crate) pos_distance: f32,
    /// Distance from the negative center of interest
    pub(crate) neg_distance: f32,
}

pub(crate) trait GetCenterOfInterestComponent {
    fn center_of_interest(&self) -> &CenterOfInterestComponent;
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub(crate) struct ContextComponent {
    pub(crate) context_value: f32,
}

pub(crate) trait GetContextComponent {
    fn context(&self) -> &ContextComponent;
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub(crate) struct MabComponent {
    pub(crate) rank: usize,
}

pub(crate) trait GetMabComponent {
    fn mab(&self) -> &MabComponent;
}

// States definition
// The transation order is:
// WithDocument -> WithEmbedding -> WithCenterOfInterest ->
// -> WithLtr -> WithContext -> WithMab

pub(crate) struct WithDocument {
    document: DocumentComponent,
}

impl GetDocumentComponent for WithDocument {
    fn document(&self) -> &DocumentComponent {
        &self.document
    }
}

pub(crate) struct WithEmbedding {
    prev_state: WithDocument,
    embedding: EmbeddingComponent,
}

impl GetEmbeddingComponent for WithEmbedding {
    fn embedding(&self) -> &EmbeddingComponent {
        &self.embedding
    }
}

pub(crate) struct WithCenterOfInterest {
    prev_state: WithEmbedding,
    center_of_interest: CenterOfInterestComponent,
}

impl GetCenterOfInterestComponent for WithCenterOfInterest {
    fn center_of_interest(&self) -> &CenterOfInterestComponent {
        &self.center_of_interest
    }
}

pub(crate) struct WithLtr {
    prev_state: WithCenterOfInterest,
    ltr: LtrComponent,
}

impl GetLtrComponent for WithLtr {
    fn ltr(&self) -> &LtrComponent {
        &self.ltr
    }
}

pub(crate) struct WithContext {
    prev_state: WithLtr,
    context: ContextComponent,
}

impl GetContextComponent for WithContext {
    fn context(&self) -> &ContextComponent {
        &self.context
    }
}

pub(crate) struct WithMab {
    prev_state: WithContext,
    mab: MabComponent,
}

impl GetMabComponent for WithMab {
    fn mab(&self) -> &MabComponent {
        &self.mab
    }
}

pub(crate) struct DocumentDataState<S> {
    inner: S,
}

pub(crate) enum DocumentData {
    Empty,
    WithEmbedding(DocumentDataState<WithEmbedding>),
    WithMab(DocumentDataState<WithMab>),
}

/// Implements the transition from state `$from` to state `$to`
macro_rules! impl_add_component {
    ($from:ty, $to: ident, $method:ident, $field:ident, $component:ty) => {
        impl DocumentDataState<$from> {
            pub(crate) fn $method(self, $field: $component) -> DocumentDataState<$to> {
                let inner = $to {
                    prev_state: self.inner,
                    $field,
                };

                DocumentDataState::<$to> { inner }
            }
        }
    };
}

/// Implements a trait to get a component that is present in a previous state
macro_rules! impl_get_component {
    ($get_component_trait:ident, $method:ident, $component:ty $(, $state:ty)*) => {
        // implement the trait for every inner state
        $(
        impl $get_component_trait for $state {
            #[inline(always)]
            fn $method(&self) -> &$component {
                &self.prev_state.$method()
            }
        }
        )*
        // implement the trait for DocumentData
        impl<S> $get_component_trait for DocumentDataState<S>
        where S: $get_component_trait
        {
            #[inline(always)]
            fn $method(&self) -> &$component {
                &self.inner.$method()
            }
        }

    };
}

impl DocumentDataState<WithDocument> {
    pub(crate) fn new(document: DocumentComponent) -> Self {
        let inner = WithDocument { document };

        Self { inner }
    }
}

impl_add_component!(
    WithDocument,
    WithEmbedding,
    add_embedding,
    embedding,
    EmbeddingComponent
);
impl_add_component!(
    WithEmbedding,
    WithCenterOfInterest,
    add_center_of_interest,
    center_of_interest,
    CenterOfInterestComponent
);
impl_add_component!(WithCenterOfInterest, WithLtr, add_ltr, ltr, LtrComponent);
impl_add_component!(WithLtr, WithContext, add_context, context, ContextComponent);
impl_add_component!(WithContext, WithMab, add_mab, mab, MabComponent);

impl_get_component!(
    GetDocumentComponent,
    document,
    DocumentComponent,
    WithEmbedding,
    WithCenterOfInterest,
    WithLtr,
    WithContext,
    WithMab
);
impl_get_component!(
    GetEmbeddingComponent,
    embedding,
    EmbeddingComponent,
    WithCenterOfInterest,
    WithLtr,
    WithContext,
    WithMab
);
impl_get_component!(
    GetCenterOfInterestComponent,
    center_of_interest,
    CenterOfInterestComponent,
    WithLtr,
    WithContext,
    WithMab
);
impl_get_component!(GetLtrComponent, ltr, LtrComponent, WithContext, WithMab);
impl_get_component!(GetContextComponent, context, ContextComponent, WithMab);
impl_get_component!(GetMabComponent, mab, MabComponent);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_and_get() {
        let document_component = DocumentComponent {
            id: DocumentId("id".to_string()),
            snippet: "snippet".to_string(),
        };
        let document_data = DocumentDataState::<WithDocument>::new(document_component.clone());
        assert_eq!(&document_component, document_data.document());

        let embedding_component = EmbeddingComponent {
            embedding: vec![1., 2., 3., 4.],
        };
        let document_data = document_data.add_embedding(embedding_component.clone());
        assert_eq!(&document_component, document_data.document());
        assert_eq!(&embedding_component, document_data.embedding());

        let coi_component = CenterOfInterestComponent {
            id: CenterOfInterestId(9),
            pos_distance: 0.7,
            neg_distance: 0.2,
        };
        let document_data = document_data.add_center_of_interest(coi_component.clone());
        assert_eq!(&document_component, document_data.document());
        assert_eq!(&embedding_component, document_data.embedding());
        assert_eq!(&coi_component, document_data.center_of_interest());

        let ltr_component = LtrComponent { context_value: 0.3 };
        let document_data = document_data.add_ltr(ltr_component.clone());
        assert_eq!(&document_component, document_data.document());
        assert_eq!(&embedding_component, document_data.embedding());
        assert_eq!(&coi_component, document_data.center_of_interest());
        assert_eq!(&ltr_component, document_data.ltr());

        let context_component = ContextComponent {
            context_value: 1.23,
        };
        let document_data = document_data.add_context(context_component.clone());
        assert_eq!(&document_component, document_data.document());
        assert_eq!(&ltr_component, document_data.ltr());
        assert_eq!(&embedding_component, document_data.embedding());
        assert_eq!(&coi_component, document_data.center_of_interest());
        assert_eq!(&context_component, document_data.context());

        let mab_component = MabComponent { rank: 3 };
        let document_data = document_data.add_mab(mab_component.clone());
        assert_eq!(&document_component, document_data.document());
        assert_eq!(&ltr_component, document_data.ltr());
        assert_eq!(&embedding_component, document_data.embedding());
        assert_eq!(&coi_component, document_data.center_of_interest());
        assert_eq!(&context_component, document_data.context());
        assert_eq!(&mab_component, document_data.mab());
    }
}
