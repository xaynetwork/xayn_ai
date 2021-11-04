import 'dart:math' show min, max;

/// Maximum number of threads to be used for multithreaded features.
const int maxNumberOfThreads = 16;

/// Selects the number of threads used by the [`XaynAi`] thread pool.
///
/// On a single core system the thread pool consists of only one thread.
/// On a multicore system the thread pool consists of
/// (the number of logical cores - 1) threads, but at most [`maxNumberOfThreads`]
/// threads and at least one thread.
int selectThreadPoolSize(int numberOfProcessors) =>
    min(max(numberOfProcessors - 1, 1), maxNumberOfThreads);
