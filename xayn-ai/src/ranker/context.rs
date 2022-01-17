use crate::{data::document_data::CoiComponent, ranker::utils::Id};

/// The context used to calculate a document's score.
/// <https://xainag.atlassian.net/wiki/spaces/M2D/pages/770670607/Production+AI+Workflow#3.2-Context-calculations>.
/// outlines the calculation of positive and negative distance factor.
pub(crate) struct Context {
    /// Average positive distance.
    pos_avg: f32,
    /// Maximum negative distance.
    neg_max: f32,
}

impl Context {
    pub(crate) fn from_cois(cois: &[(Id, CoiComponent)]) -> Self {
        let cois_len = cois.len() as f32;
        let pos_avg = cois.iter().map(|(_, coi)| coi.pos_distance).sum::<f32>() / cois_len;
        let neg_max = cois
            .iter()
            .map(|(_, coi)| coi.neg_distance)
            .fold(f32::MIN, f32::max); // NOTE f32::max considers NaN as smallest value

        Self { pos_avg, neg_max }
    }

    /// Calculates score from given positive distance and negative distance.
    /// Both positive and negative distance must be >= 0.
    pub(crate) fn calculate_score(&self, pos: f32, neg: f32) -> f32 {
        debug_assert!(pos >= 0. && neg >= 0.);
        let frac_pos = (self.pos_avg > 0.)
            .then(|| (1. + pos / self.pos_avg).recip())
            .unwrap_or(1.);
        let frac_neg = (1. + (self.neg_max - neg)).recip();

        (frac_pos + frac_neg) / 2.
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        data::document_data::CoiComponent,
        ranker::{context::Context, utils::Id},
        CoiId,
    };
    use test_utils::assert_approx_eq;

    #[allow(clippy::eq_op)]
    #[test]
    fn test_calculate() {
        let calc = Context {
            pos_avg: 4.,
            neg_max: 8.,
        };

        // In the `assert_approx_eq!`s below, the result expectations are
        // expressions instead of constants to make it easier to understand how
        // the values come into existence. The format is (frac_pos + frac_neg) / 2.

        let cxt = calc.calculate_score(0., calc.neg_max);
        assert_approx_eq!(f32, cxt, (1. + 1.) / 2.);

        let cxt = calc.calculate_score(1., calc.neg_max);
        assert_approx_eq!(f32, cxt, ((4. / 5.) + 1.) / 2.);

        let cxt = calc.calculate_score(calc.pos_avg, calc.neg_max);
        assert_approx_eq!(f32, cxt, (1. / 2. + 1.) / 2.);

        let cxt = calc.calculate_score(8., 7.);
        assert_approx_eq!(f32, cxt, ((1. / 3.) + (1. / 2.)) / 2.);
    }

    #[allow(clippy::eq_op)]
    #[test]
    fn test_calculate_neg_max_f32_max() {
        // when calculating the negative distance in the `CoiSystem`,
        // we assign `f32::MAX` if we don't have negative cois
        let calc = Context {
            pos_avg: 4.,
            neg_max: f32::MAX,
        };

        let ctx = calc.calculate_score(0., calc.neg_max);
        assert_approx_eq!(f32, ctx, (1. + 1.) / 2.);
    }

    #[test]
    fn test_from_cois() {
        let cois = vec![
            (
                Id::from_u128(0),
                CoiComponent {
                    id: CoiId::mocked(1),
                    pos_distance: 1.,
                    neg_distance: 10.,
                },
            ),
            (
                Id::from_u128(0),
                CoiComponent {
                    id: CoiId::mocked(2),
                    pos_distance: 6.,
                    neg_distance: 4.,
                },
            ),
            (
                Id::from_u128(0),
                CoiComponent {
                    id: CoiId::mocked(3),
                    pos_distance: 8.,
                    neg_distance: 2.,
                },
            ),
        ];

        let calc = Context::from_cois(cois.as_slice());
        assert_approx_eq!(f32, calc.pos_avg, 5.);
        assert_approx_eq!(f32, calc.neg_max, 10.);
    }
}
