use std::{marker::PhantomData, ptr::null, slice::from_raw_parts};

use ffi_support::ExternError;

#[repr(C)]
pub struct Key<'a> {
    _a: PhantomData<&'a ()>,
    /// The raw pointer to the data.
    pub data: *const u8,
    /// The length of the data.
    pub len: u32,
}

impl<'a, K> From<K> for Key<'a>
where
    K: AsRef<[u8]> + 'a,
{
    fn from(key: K) -> Self {
        let key = key.as_ref();
        Self {
            _a: PhantomData,
            data: key.as_ptr(),
            len: key.len() as u32,
        }
    }
}

#[repr(C)]
pub struct Value<'a> {
    _a: PhantomData<&'a ()>,
    /// The raw pointer to the data.
    pub data: *const u8,
    /// The length of the data.
    pub len: u32,
}

impl<'a, V> From<V> for Value<'a>
where
    V: AsRef<[u8]> + 'a,
{
    fn from(value: V) -> Self {
        let value = value.as_ref();
        Self {
            _a: PhantomData,
            data: value.as_ptr(),
            len: value.len() as u32,
        }
    }
}

impl Value<'_> {
    fn new() -> Self {
        Self {
            _a: PhantomData,
            data: null(),
            len: 0,
        }
    }

    unsafe fn to_vec(&self) -> Option<Vec<u8>> {
        if self.data.is_null() || self.len == 0 {
            None
        } else {
            Some(unsafe { from_raw_parts(self.data, self.len as usize) }.to_vec())
        }
    }
}

#[repr(C)]
pub struct Database {
    /// The callback to get an entry.
    pub get: unsafe extern "C" fn(*const Key, *mut ExternError) -> *mut Value,
    /// The callback to insert an entry.
    pub insert: unsafe extern "C" fn(*const Key, *const Value, *mut ExternError),
    /// The callback to delete an entry.
    pub delete: unsafe extern "C" fn(*const Key, *mut ExternError),
    /// The callback to free the memory of the gotten entry value.
    pub drop_value: unsafe extern "C" fn(*mut Value),
    /// The callback to free the memory of the error message.
    pub drop_msg: unsafe extern "C" fn(*mut ExternError),
}
