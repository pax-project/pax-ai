//! Blocking work (a `nix` call, `pdftotext`) run off the async executor.

use std::time::Duration;

/// Runs blocking work on tokio's blocking pool, so it never stalls the
/// executor; `None` once `timeout` passes. The work itself can't be
/// cancelled and runs on to its end in the background.
pub(crate) async fn blocking_with_timeout<T: Send + 'static>(
    timeout: Duration,
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<Option<T>, String> {
    match tokio::time::timeout(timeout, tokio::task::spawn_blocking(work)).await {
        Ok(joined) => joined
            .map(Some)
            .map_err(|e| format!("the task failed: {e}")),
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn blocking_work_past_its_timeout_is_reported_as_none() {
        let slow = blocking_with_timeout(Duration::from_millis(10), || {
            std::thread::sleep(Duration::from_millis(500));
        })
        .await
        .unwrap();
        assert!(slow.is_none());
        let quick = blocking_with_timeout(Duration::from_secs(5), || 42)
            .await
            .unwrap();
        assert_eq!(quick, Some(42));
    }
}
