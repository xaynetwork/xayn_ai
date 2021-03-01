pub mod string;

use unicode_categories::UnicodeCategories;

use crate::normalizer::string::NormalizedString;

/// A Bert normalizer.
pub struct Normalizer {
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: bool,
    lowercase: bool,
}

impl Normalizer {
    pub(crate) fn new(
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: bool,
        lowercase: bool,
    ) -> Self {
        Self {
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
        }
    }

    fn clean_text(&self, normalized: NormalizedString) -> NormalizedString {
        if self.clean_text {
            normalized
                .filter(|c| {
                    c != '\0'
                        && c != '\u{fffd}'
                        && (c == '\t' || c == '\n' || c == '\r' || !c.is_other())
                })
                .map(|c| {
                    // These are technically control characters but we count them as whitespace
                    // The definition of `is_control` here is quite large and contains also
                    // Cc, Cf, Cn or Co; cf. https://unicode.org/reports/tr44/ (Table 12)
                    if c == '\t' || c == '\n' || c == '\r' || c.is_whitespace() {
                        ' '
                    } else {
                        c
                    }
                })
        } else {
            normalized
        }
    }

    fn handle_chinese_chars(&self, normalized: NormalizedString) -> NormalizedString {
        if self.handle_chinese_chars {
            let mut new_chars: Vec<(char, isize)> = vec![];
            normalized.for_each(|c| {
                // Checks whether a character is chinese
                // This defines a "chinese character" as anything in the CJK Unicode block:
                //   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
                //
                // Note that the CJK Unicode block is NOT all Japanese and Korean characters,
                // despite its name. The modern Korean Hangul alphabet is a different block,
                // as is Japanese Hiragana and Katakana. Those alphabets are used to write
                // space-separated words, so they are not treated specially and handled
                // like for all of the other languages.
                if let '\u{4E00}'..='\u{9FFF}'
                | '\u{3400}'..='\u{4DBF}'
                | '\u{20000}'..='\u{2A6DF}'
                | '\u{2A700}'..='\u{2B73F}'
                | '\u{2B740}'..='\u{2B81F}'
                | '\u{2B920}'..='\u{2CEAF}'
                | '\u{F900}'..='\u{FAFF}'
                | '\u{2F800}'..='\u{2FA1F}' = c
                {
                    new_chars.extend(&[(' ', 0), (c, 1), (' ', 1)]);
                } else {
                    new_chars.push((c, 0));
                }
            });
            normalized.transform(new_chars, 0)
        } else {
            normalized
        }
    }

    fn strip_accents(&self, normalized: NormalizedString) -> NormalizedString {
        if self.strip_accents {
            normalized.nfd().filter(|c| !c.is_mark_nonspacing())
        } else {
            normalized
        }
    }

    fn lowercase(&self, normalized: NormalizedString) -> NormalizedString {
        if self.lowercase {
            normalized.lowercase()
        } else {
            normalized
        }
    }

    pub(crate) fn normalize(&self, sequence: impl AsRef<str>) -> NormalizedString {
        let normalized = sequence.into();
        let normalized = self.clean_text(normalized);
        let normalized = self.handle_chinese_chars(normalized);
        let normalized = self.strip_accents(normalized);
        self.lowercase(normalized)
    }
}
