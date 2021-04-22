use wasm_bindgen::JsValue;

pub trait IntoJsResult<T> {
    fn into_js_result(self) -> Result<T, JsValue>;
}

impl<T, E> ToJsResult<T> for Result<T, E>
where
    E: std::fmt::Debug,
{
    fn to_js_result(self) -> Result<T, JsValue> {
        self.map_err(|e| JsValue::from_str(&format!("{:#?}", e)))
    }
}
