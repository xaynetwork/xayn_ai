use wasm_bindgen::JsValue;
use xayn_ai_ffi::Error;

pub trait IntoJsResult<T> {
    fn into_js_result(self) -> Result<T, JsValue>;
}

impl<T> IntoJsResult<T> for Result<T, Error> {
    fn into_js_result(self) -> Result<T, JsValue> {
        self.map_err(|e| JsValue::from_serde(&e).expect("Failed to serialize the error"))
    }
}
