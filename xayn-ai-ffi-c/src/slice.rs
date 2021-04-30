use std::{
    fmt,
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
    slice,
};

#[repr(C)]
pub struct CBoxedSlice<T: Sized> {
    data: Option<NonNull<T>>,
    len: u64,
    _owned: PhantomData<T>,
}

unsafe fn into_boxed_slice_unchecked<T: Sized>(data: NonNull<T>, len: u64) -> Box<[T]> {
    let raw_slice = ptr::slice_from_raw_parts_mut(data.as_ptr(), len as usize);
    unsafe { Box::from_raw(raw_slice) }
}

impl<T> Drop for CBoxedSlice<T> {
    fn drop(&mut self) {
        if self.is_sound() {
            if let Some(data) = self.data {
                unsafe { into_boxed_slice_unchecked(data, self.len) };
            }
        }
    }
}

impl<T> CBoxedSlice<T> {
    pub fn new(boxed_slice: Box<[T]>) -> Self {
        let len = boxed_slice.len() as u64;
        let data = NonNull::new(Box::leak(boxed_slice).as_mut_ptr());
        let _owned = PhantomData;

        Self { data, len, _owned }
    }

    // aligned and addressable; or empty
    pub fn is_sound(&self) -> bool {
        if let Some(data) = self.data {
            data.as_ptr() as usize % mem::align_of::<T>() == 0
                && (mem::size_of::<T>() as u64).saturating_mul(self.len) <= isize::MAX as u64
        } else {
            self.len == 0
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.data
            .map(|data| unsafe {
                slice::from_raw_parts(data.as_ptr() as *const T, self.len as usize)
            })
            .unwrap_or_default()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
            .map(|data| unsafe { slice::from_raw_parts_mut(data.as_ptr(), self.len as usize) })
            .unwrap_or_default()
    }

    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.data
            .map(|data| unsafe { into_boxed_slice_unchecked(data, self.len) })
            .unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> From<Box<[T]>> for CBoxedSlice<T> {
    fn from(boxed_slice: Box<[T]>) -> Self {
        Self::new(boxed_slice)
    }
}

impl<T> From<CBoxedSlice<T>> for Box<[T]> {
    fn from(boxed_slice: CBoxedSlice<T>) -> Self {
        boxed_slice.into_boxed_slice()
    }
}

impl<T> Deref for CBoxedSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for CBoxedSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> AsRef<[T]> for CBoxedSlice<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for CBoxedSlice<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: fmt::Debug> fmt::Debug for CBoxedSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl<T> fmt::Pointer for CBoxedSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(
            &self.data.map(NonNull::as_ptr).unwrap_or_else(ptr::null_mut),
            f,
        )
    }
}

// owned, unaliased
unsafe impl<T: Send> Send for CBoxedSlice<T> {}

// owned, unaliased
unsafe impl<T: Sync> Sync for CBoxedSlice<T> {}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::utils::tests::{as_str_unchecked, AsPtr};

    impl CBoxedSlice<u8> {
        /// See [`as_str_unchecked()`] for more.
        pub fn as_str_unchecked(&self) -> &str {
            as_str_unchecked(self.as_slice().first())
        }
    }

    impl<T> AsPtr for CBoxedSlice<T> {}
}
