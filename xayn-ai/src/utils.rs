
#[macro_export]
macro_rules! to_vec_of_ref_of {
	  ($data: expr, $type:ty) => {
		   $data.iter().map(|data| -> $type { data }).collect::<Vec<_>>()
	  };
}
