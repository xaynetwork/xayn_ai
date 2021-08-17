use indicatif::{ProgressBar, ProgressStyle};

pub(crate) fn progress_spin_until_done<R>(msg: &'static str, func: impl FnOnce() -> R) -> R {
    let progress_bar = ProgressBar::new_spinner()
        .with_style(ProgressStyle::default_bar().template("{msg}: {elapsed:>10} {spinner:.green}"));
    progress_bar.set_message(msg);
    progress_bar.enable_steady_tick(100);
    let res = func();
    progress_bar.finish();
    res
}
