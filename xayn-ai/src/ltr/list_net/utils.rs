#[cfg(test)]
macro_rules! assert_ndarray_eq {
    ($t:ty, $left:expr, $right:expr) => (assert_ndarray_eq!($t, $left, $right, ulps = 2));
    ($t:ty, $left:expr, $right:expr, ulps = $ulps:expr) => ({
        use ::itertools::izip;
        use ::float_cmp::approx_eq;
        use ::ndarray::Axis;

        let left = $left;
        let right = $right;
        let ulps = $ulps;

        if left.shape() != right.shape() {
            panic!(concat!(
                "Cannot compare arrays. Array shape mismatch: {sl:?} != {sr:?}",
                "\nLeft array: {la:?}",
                "\nRight array: {ra:?}"
            ), sl=left.shape(), sr=right.shape(), la=&left, ra=&right);
        }
        // shape is equals so left.ndim() == right.ndim()
        let ndim = left.ndim();
        if ndim > 0 {
            let left_lanes = left.lanes(Axis(ndim-1));
            let right_lanes = right.lanes(Axis(ndim-1));
            for (lane_idx, (left_lane, right_lane)) in izip!(left_lanes, right_lanes).enumerate() {
                for (left_elem, right_elem) in izip!(left_lane, right_lane) {
                    if !approx_eq!($t, *left_elem, *right_elem, ulps = $ulps) {
                        panic!(concat!(
                            "Arrays are not equal (ulps={ulps}): First mismatch in the idx={lane_idx} inner most lane: {l:?} != {r:?}",
                            "\nLeft array: {la:?}",
                            "\nRight array: {ra:?}"
                        ), ulps=ulps, lane_idx=lane_idx, l=left_elem, r=right_elem, la=&left, ra=&right);
                    }
                }
            }
        }
    });
}
