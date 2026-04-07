from dataclasses import dataclass, field
from threading import Lock


@dataclass
class RuntimeMetrics:
    prediction_requests: int = 0
    batch_requests: int = 0
    pdf_parse_requests: int = 0
    _lock: Lock = field(default_factory=Lock)

    def increment_predictions(self) -> None:
        with self._lock:
            self.prediction_requests += 1

    def increment_batch(self) -> None:
        with self._lock:
            self.batch_requests += 1

    def increment_pdf(self) -> None:
        with self._lock:
            self.pdf_parse_requests += 1


runtime_metrics = RuntimeMetrics()

