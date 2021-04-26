use wasm_bindgen::JsValue;

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
