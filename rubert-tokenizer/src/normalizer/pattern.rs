use regex::Regex;

use crate::normalizer::string::Offsets;

/// A pattern used to split a NormalizedString.
pub trait Pattern {
    /// Slices the sequence in a list of pattern match positions.
    ///
    /// A boolean indicates whether this is a match or not. This method *must* cover the whole
    /// string in its outputs, with contiguous ordered slices.
    fn find_matches(&self, inside: &str) -> Vec<(Offsets, bool)>;
}

impl<F> Pattern for F
where
    F: Fn(char) -> bool,
{
    fn find_matches(&self, inside: &str) -> Vec<(Offsets, bool)> {
        if inside.is_empty() {
            return vec![(Offsets(0, 0), false)];
        }

        let mut last_offset = 0;
        let mut last_seen = 0;

        let mut matches = inside
            .char_indices()
            .flat_map(|(idx, chr)| {
                last_seen = idx + chr.len_utf8();
                if self(chr) {
                    let mut events = Vec::with_capacity(2);
                    if last_offset < idx {
                        // We need to emit what was before this match
                        events.push((Offsets(last_offset, idx), false));
                    }
                    events.push((Offsets(idx, last_seen), true));
                    last_offset = last_seen;
                    events
                } else {
                    vec![]
                }
            })
            .collect::<Vec<_>>();

        // Do not forget the last potential split
        if last_seen > last_offset {
            matches.push((Offsets(last_offset, last_seen), false));
        }

        matches
    }
}

impl Pattern for char {
    fn find_matches(&self, inside: &str) -> Vec<(Offsets, bool)> {
        let is_char = |c: char| -> bool { c == *self };
        is_char.find_matches(inside)
    }
}

impl Pattern for str {
    fn find_matches(&self, inside: &str) -> Vec<(Offsets, bool)> {
        if self.is_empty() {
            return vec![(Offsets(0, inside.chars().count()), false)];
        }

        Regex::new(regex::escape(self).as_str())
            .expect("escape regex is infallible")
            .find_matches(inside)
    }
}

impl Pattern for String {
    fn find_matches(&self, inside: &str) -> Vec<(Offsets, bool)> {
        self.as_str().find_matches(inside)
    }
}

impl Pattern for Regex {
    fn find_matches(&self, inside: &str) -> Vec<(Offsets, bool)> {
        if inside.is_empty() {
            return vec![(Offsets(0, 0), false)];
        }

        let mut prev = 0;
        let mut matches = Vec::with_capacity(inside.len());
        for m in self.find_iter(inside) {
            if prev != m.start() {
                matches.push((Offsets(prev, m.start()), false));
            }
            matches.push((Offsets(m.start(), m.end()), true));
            prev = m.end();
        }
        if prev != inside.len() {
            matches.push((Offsets(prev, inside.len()), false))
        }

        matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_functions() {
        let is_b = |c| c == 'b';
        assert_eq!(
            is_b.find_matches("aba"),
            vec![
                (Offsets(0, 1), false),
                (Offsets(1, 2), true),
                (Offsets(2, 3), false),
            ],
        );
        assert_eq!(
            is_b.find_matches("aaaab"),
            vec![(Offsets(0, 4), false), (Offsets(4, 5), true)],
        );
        assert_eq!(
            is_b.find_matches("bbaaa"),
            vec![
                (Offsets(0, 1), true),
                (Offsets(1, 2), true),
                (Offsets(2, 5), false),
            ],
        );
        assert_eq!(is_b.find_matches(""), vec![(Offsets(0, 0), false)]);
        assert_eq!(is_b.find_matches("aaa"), vec![(Offsets(0, 3), false)]);
    }

    #[test]
    fn test_char() {
        assert_eq!(
            'a'.find_matches("aaa"),
            vec![
                (Offsets(0, 1), true),
                (Offsets(1, 2), true),
                (Offsets(2, 3), true),
            ],
        );
        assert_eq!(
            'a'.find_matches("bbbba"),
            vec![(Offsets(0, 4), false), (Offsets(4, 5), true)],
        );
        assert_eq!(
            'a'.find_matches("aabbb"),
            vec![
                (Offsets(0, 1), true),
                (Offsets(1, 2), true),
                (Offsets(2, 5), false),
            ],
        );
        assert_eq!('a'.find_matches(""), vec![(Offsets(0, 0), false)]);
        assert_eq!('b'.find_matches("aaa"), vec![(Offsets(0, 3), false)]);
    }

    #[test]
    fn test_str() {
        assert_eq!(
            "a".find_matches("aba"),
            vec![
                (Offsets(0, 1), true),
                (Offsets(1, 2), false),
                (Offsets(2, 3), true),
            ],
        );
        assert_eq!(
            "a".find_matches("bbbba"),
            vec![(Offsets(0, 4), false), (Offsets(4, 5), true)],
        );
        assert_eq!(
            "a".find_matches("aabbb"),
            vec![
                (Offsets(0, 1), true),
                (Offsets(1, 2), true),
                (Offsets(2, 5), false),
            ],
        );
        assert_eq!(
            "ab".find_matches("aabbb"),
            vec![
                (Offsets(0, 1), false),
                (Offsets(1, 3), true),
                (Offsets(3, 5), false),
            ],
        );
        assert_eq!(
            "ab".find_matches("aabbab"),
            vec![
                (Offsets(0, 1), false),
                (Offsets(1, 3), true),
                (Offsets(3, 4), false),
                (Offsets(4, 6), true),
            ],
        );
        assert_eq!("".find_matches(""), vec![(Offsets(0, 0), false)]);
        assert_eq!("".find_matches("aaa"), vec![(Offsets(0, 3), false)]);
        assert_eq!("b".find_matches("aaa"), vec![(Offsets(0, 3), false)]);
    }

    #[test]
    fn test_regex() {
        let is_whitespace = Regex::new(r"\s+").unwrap();
        assert_eq!(
            is_whitespace.find_matches("a   b"),
            vec![
                (Offsets(0, 1), false),
                (Offsets(1, 4), true),
                (Offsets(4, 5), false),
            ],
        );
        assert_eq!(
            is_whitespace.find_matches("   a   b   "),
            vec![
                (Offsets(0, 3), true),
                (Offsets(3, 4), false),
                (Offsets(4, 7), true),
                (Offsets(7, 8), false),
                (Offsets(8, 11), true),
            ],
        );
        assert_eq!(is_whitespace.find_matches(""), vec![(Offsets(0, 0), false)],);
        assert_eq!(
            is_whitespace.find_matches("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘"),
            vec![
                (Offsets(0, 16), false),
                (Offsets(16, 17), true),
                (Offsets(17, 45), false),
            ],
        );
        assert_eq!(
            is_whitespace.find_matches("aaa"),
            vec![(Offsets(0, 3), false)],
        );
    }
}
