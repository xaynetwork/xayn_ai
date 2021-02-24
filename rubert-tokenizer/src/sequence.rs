use std::borrow::Cow;

pub enum Sequence<'s> {
    Raw(Cow<'s, str>),
    PreTokenized(Cow<'s, [&'s str]>),
    PreTokenizedOwned(Cow<'s, [String]>),
    PreTokenizedCow(Cow<'s, [Cow<'s, str>]>),
}

impl<'s> From<&'s str> for Sequence<'s> {
    fn from(input: &'s str) -> Self {
        Self::Raw(Cow::Borrowed(input))
    }
}

impl From<String> for Sequence<'_> {
    fn from(input: String) -> Self {
        Self::Raw(Cow::Owned(input))
    }
}

impl<'s> From<Cow<'s, str>> for Sequence<'s> {
    fn from(input: Cow<'s, str>) -> Self {
        Self::Raw(input)
    }
}

impl<'s> From<&'s [&'s str]> for Sequence<'s> {
    fn from(input: &'s [&'s str]) -> Self {
        Self::PreTokenized(Cow::Borrowed(input))
    }
}

impl<'s> From<Vec<&'s str>> for Sequence<'s> {
    fn from(input: Vec<&'s str>) -> Self {
        Self::PreTokenized(Cow::Owned(input))
    }
}

impl<'s> From<Cow<'s, [&'s str]>> for Sequence<'s> {
    fn from(input: Cow<'s, [&'s str]>) -> Self {
        Self::PreTokenized(input)
    }
}

impl<'s> From<&'s [String]> for Sequence<'s> {
    fn from(input: &'s [String]) -> Self {
        Self::PreTokenizedOwned(Cow::Borrowed(input))
    }
}

impl<'s> From<Vec<String>> for Sequence<'s> {
    fn from(input: Vec<String>) -> Self {
        Self::PreTokenizedOwned(Cow::Owned(input))
    }
}

impl<'s> From<Cow<'s, [String]>> for Sequence<'s> {
    fn from(input: Cow<'s, [String]>) -> Self {
        Self::PreTokenizedOwned(input)
    }
}

impl<'s> From<&'s [Cow<'s, str>]> for Sequence<'s> {
    fn from(input: &'s [Cow<'s, str>]) -> Self {
        Self::PreTokenizedCow(Cow::Borrowed(input))
    }
}

impl<'s> From<Vec<Cow<'s, str>>> for Sequence<'s> {
    fn from(input: Vec<Cow<'s, str>>) -> Self {
        Self::PreTokenizedCow(Cow::Owned(input))
    }
}

impl<'s> From<Cow<'s, [Cow<'s, str>]>> for Sequence<'s> {
    fn from(input: Cow<'s, [Cow<'s, str>]>) -> Self {
        Self::PreTokenizedCow(input)
    }
}
