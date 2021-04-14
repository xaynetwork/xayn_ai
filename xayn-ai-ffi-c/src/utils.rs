/// This function does nothing.
///
/// Calling this prevents Swift to drop the library.
#[no_mangle]
pub extern "C" fn dummy_function() {}

#[cfg(test)]
pub(crate) mod tests {
    /// Common casts from references to pointers.
    ///
    /// By default, a im/mutable reference to `Self` is cast as a im/mutable pointer to `Self`. In
    /// addition, the target type `T` can be changed as well.
    ///
    /// # Safety
    /// The cast itself is safe, although it is unsafe to use the resulting pointer. The behavior is
    /// undefined if:
    /// - A `T` different from `Self` doesn't have the same memory layout.
    /// - A pointer is accessed after the lifetime of the corresponding reference ends.
    /// - A pointer of an immutable reference is accessed mutably.
    pub trait AsPtr<'a, T = Self>
    where
        Self: 'a,
    {
        /// Casts the immutable reference as a constant pointer.
        #[inline]
        fn as_ptr(&self) -> *const T {
            self as *const Self as *const T
        }

        /// Casts the mutable reference as a mutable pointer.
        #[inline]
        fn as_mut_ptr(&mut self) -> *mut T {
            self as *mut Self as *mut T
        }
    }
}
