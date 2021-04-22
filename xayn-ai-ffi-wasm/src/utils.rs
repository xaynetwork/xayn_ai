use serde::Serialize;
use wasm_bindgen::JsValue;

#[derive(Serialize)]
pub struct ExternError {
    pub code: i32,
    pub msg: String,
}

pub trait IntoJsResult<T> {
    fn into_js_result(self) -> Result<T, JsValue>;
}

impl<T, E> IntoJsResult<T> for Result<T, E>
where
    E: std::fmt::Display,
{
    fn into_js_result(self) -> Result<T, JsValue> {
        self.map_err(|e| JsValue::from_str(&format!("{}", e)))
    }
}

impl<T> IntoJsResult<T> for Result<T, ExternError> {
    fn into_js_result(self) -> Result<T, JsValue> {
        self.map_err(|e| JsValue::from_serde(&e).unwrap())
    }
}
