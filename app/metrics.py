# app/metrics.py
from __future__ import annotations
import time
import threading
from collections import defaultdict
from typing import Dict, List

class _Histogram:
    # fixed buckets in ms (log-spaced)
    BUCKETS = [10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000]
    def __init__(self):
        self.counts = [0]*len(self.BUCKETS)
        self.lock = threading.Lock()
        self._samples: List[float] = []  # keep short tail to approximate p95
    def observe_ms(self, ms: float):
        i = 0
        while i < len(self.BUCKETS) and ms > self.BUCKETS[i]:
            i += 1
        with self.lock:
            if i < len(self.BUCKETS):
                self.counts[i] += 1
            else:
                # overflow
                self.counts[-1] += 1
            # keep a rolling window of last 200 samples for p95
            self._samples.append(ms)
            if len(self._samples) > 200:
                self._samples = self._samples[-200:]
    def p95_ms(self) -> float:
        with self.lock:
            if not self._samples:
                return 0.0
            arr = sorted(self._samples)
            idx = int(0.95 * (len(arr)-1))
            return arr[idx]

class Metrics:
    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.histos: Dict[str, _Histogram] = defaultdict(_Histogram)
        self.lock = threading.Lock()
        self.process_start_ns = time.time_ns()
    def inc(self, key: str, n: int = 1):
        with self.lock:
            self.counters[key] += n
    def observe_ms(self, key: str, ms: float):
        self.histos[key].observe_ms(ms)
    def snapshot(self) -> Dict:
        up_ms = (time.time_ns() - self.process_start_ns) / 1e6
        out = {"uptime_ms": up_ms, "counters": dict(self.counters), "latency_p95_ms": {}}
        for k, h in self.histos.items():
            out["latency_p95_ms"][k] = h.p95_ms()
        return out

metrics = Metrics()
