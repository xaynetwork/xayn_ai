use std::{marker::PhantomData, slice::from_raw_parts};

use ffi_support::ExternError;
use xayn_ai::{DatabaseRaw, Error};

use crate::{error::CError, utils::AsPtr};

#[repr(C)]
pub struct CKey<'a> {
    _a: PhantomData<&'a ()>,
    /// The raw pointer to the data.
    pub data: *const u8,
    /// The length of the data.
    pub len: u32,
}

impl AsPtr for CKey<'_> {}

impl<'a, K> From<K> for CKey<'a>
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
pub struct CValue<'a> {
    _a: PhantomData<&'a ()>,
    /// The raw pointer to the data.
    pub data: *const u8,
    /// The length of the data.
    pub len: u32,
}

impl AsPtr for CValue<'_> {}

impl<'a, V> From<V> for CValue<'a>
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

impl CValue<'_> {
    unsafe fn to_vec(&self) -> Option<Vec<u8>> {
        if self.data.is_null() || self.len == 0 {
            None
        } else {
            Some(unsafe { from_raw_parts(self.data, self.len as usize) }.to_vec())
        }
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct CDatabase {
    /// The callback to get an entry.
    pub get: unsafe extern "C" fn(*const CKey, *mut ExternError) -> *mut CValue,
    /// The callback to insert an entry.
    pub insert: unsafe extern "C" fn(*const CKey, *const CValue, *mut ExternError),
    /// The callback to delete an entry.
    pub delete: unsafe extern "C" fn(*const CKey, *mut ExternError),
    /// The callback to free the memory of the gotten entry value.
    pub drop_value: unsafe extern "C" fn(*mut CValue),
    /// The callback to free the memory of the error message.
    pub drop_msg: unsafe extern "C" fn(*mut ExternError),
}

impl CDatabase {
    unsafe fn drop_value(&self, value: *mut CValue) {
        if let Some(value) = unsafe { value.as_mut() } {
            unsafe { (self.drop_value)(value) };
        }
    }

    unsafe fn drop_message(&self, error: *mut ExternError) {
        if let Some(error) = unsafe { error.as_mut() } {
            unsafe { (self.drop_msg)(error) };
        }
    }
}

impl DatabaseRaw for CDatabase {
    fn get(&self, key: impl AsRef<[u8]>) -> Result<Option<Vec<u8>>, Error> {
        let key = CKey::from(key);
        let mut error = ExternError::success();

        let value = unsafe { (self.get)(key.as_ptr(), error.as_mut_ptr()) };
        let code = error.get_code().into();
        let result = if let (CError::Success, Some(value)) = (code, unsafe { value.as_mut() }) {
            Ok(unsafe { value.to_vec() })
        } else {
            Err(code.with_anyhow_context(error.get_message().as_opt_str()))
        };

        unsafe { self.drop_value(value) };
        unsafe { self.drop_message(error.as_mut_ptr()) };
        result
    }

    fn insert(&self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Result<(), Error> {
        let key = CKey::from(key);
        let value = CValue::from(value);
        let mut error = ExternError::success();

        unsafe { (self.insert)(key.as_ptr(), value.as_ptr(), error.as_mut_ptr()) };
        let code = CError::from(error.get_code());
        let result = if let CError::Success = code {
            Ok(())
        } else {
            Err(code.with_anyhow_context(error.get_message().as_opt_str()))
        };

        unsafe { self.drop_message(error.as_mut_ptr()) };
        result
    }

    fn delete(&self, key: impl AsRef<[u8]>) -> Result<(), Error> {
        let key = CKey::from(key);
        let mut error = ExternError::success();

        unsafe { (self.delete)(key.as_ptr(), error.as_mut_ptr()) };
        let code = CError::from(error.get_code());
        let result = if let CError::Success = code {
            Ok(())
        } else {
            Err(code.with_anyhow_context(error.get_message().as_opt_str()))
        };

        unsafe { self.drop_message(error.as_mut_ptr()) };
        result
    }
}
