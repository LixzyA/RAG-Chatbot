from typing import Any


class _NoOpSpan:
    """A span that does nothing."""

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: ANN401
        pass

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: ANN401
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass


class _NoOpTracer:
    """A tracer that returns no-op spans."""

    def start_span(self, name: str, *, context: Any = None) -> _NoOpSpan:  # noqa: ANN401
        return _NoOpSpan()

    def get_current_span(self) -> _NoOpSpan:
        return _NoOpSpan()


_tracer: _NoOpTracer | None = None


def get_tracer() -> _NoOpTracer:
    """Return the global tracer instance (no-op until a real backend is configured)."""
    global _tracer  # noqa: PLW0603
    if _tracer is None:
        _tracer = _NoOpTracer()
    return _tracer
