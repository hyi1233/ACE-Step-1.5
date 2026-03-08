"""Simple in-process job queue for GPU-bound music generation.

Design goals:
- single worker (1 job at a time) to avoid GPU contention
- stable API for polling job state
- minimal dependencies (no Redis required)

This is NOT intended to be a distributed queue.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Optional

JobFn = Callable[[str, dict], dict]


@dataclass
class JobState:
    job_id: str
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    status: str = "queued"
    position: int = 0
    error: Optional[str] = None
    request: dict = field(default_factory=dict)
    result: Optional[dict] = None


class InProcessJobQueue:
    """A FIFO queue with a single worker thread.

    NOTE: Keep constructor backward-compatible across patches.
    """

    def __init__(self, worker_fn: JobFn, outputs_root: Optional[str] = None, **kwargs):
        if outputs_root is None:
            outputs_root = (
                kwargs.get("outputs_dir")
                or kwargs.get("results_root")
                or kwargs.get("results_dir")
                or kwargs.get("output_dir")
            )
        if outputs_root is None:
            outputs_root = os.path.join(os.getcwd(), "aceflow_outputs")

        self._worker_fn = worker_fn
        self._outputs_root = str(outputs_root)
        self._q: Deque[str] = deque()
        self._jobs: Dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = False
        self._running_job_id: Optional[str] = None

        os.makedirs(self._outputs_root, exist_ok=True)

        self._thread = threading.Thread(target=self._loop, name="ace-step-remote-queue", daemon=True)
        self._thread.start()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def submit(self, job_id: str, request: dict) -> JobState:
        with self._cv:
            state = JobState(job_id=job_id, request=request)
            self._jobs[job_id] = state
            self._q.append(job_id)
            self._recompute_positions_locked()
            self._cv.notify_all()
            return state

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot_queue(self) -> dict:
        with self._lock:
            return {
                "running": self._running_job_id,
                "queued": list(self._q),
                "queue_length": len(self._q),
            }

    def _recompute_positions_locked(self):
        for idx, jid in enumerate(self._q, start=1):
            st = self._jobs.get(jid)
            if st:
                st.position = idx
        if self._running_job_id and self._running_job_id in self._jobs:
            self._jobs[self._running_job_id].position = 0

    def _loop(self):
        while True:
            with self._cv:
                while not self._stop and not self._q:
                    self._cv.wait(timeout=0.5)
                if self._stop:
                    return
                job_id = self._q.popleft()
                self._running_job_id = job_id
                st = self._jobs.get(job_id)
                if st:
                    st.status = "running"
                    st.started_at = time.time()
                self._recompute_positions_locked()

            try:
                result = self._worker_fn(job_id, st.request if st else {})
                with self._lock:
                    st2 = self._jobs.get(job_id)
                    if st2:
                        st2.status = "done"
                        st2.finished_at = time.time()
                        st2.result = result
            except Exception as e:
                with self._lock:
                    st2 = self._jobs.get(job_id)
                    if st2:
                        st2.status = "error"
                        st2.finished_at = time.time()
                        st2.error = str(e)
            finally:
                with self._lock:
                    self._running_job_id = None
                    self._recompute_positions_locked()
