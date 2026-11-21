"""Run offline vLLM in a process that can be stopped on a wall-clock deadline."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import multiprocessing
import os
import signal
import subprocess
import sys

from .inference import BatchEngineUnavailable, VLLMBackend, format_error

DEFAULT_ENGINE_TIMEOUT = 1800


# Run outside the worker interpreter so native engine calls cannot delay cleanup.
_WATCH_COLLECTOR = """
import os
import select
import signal
import sys

collector, worker = map(int, sys.argv[1:])
control = sys.stdin.fileno()
poller = select.poll()
poller.register(collector, select.POLLIN)
poller.register(control, select.POLLIN)
while os.getppid() == worker:
    ready = dict(poller.poll(250))
    if collector in ready:
        break
    if control in ready:
        if os.read(control, 1) == b"q":
            sys.exit(0)
        break
os.killpg(os.getpgrp(), signal.SIGKILL)
"""


def _run_worker(connection, factory, spec, options):
    # The worker and its engine children have a dedicated group. Timing out
    # must never signal the collector or other jobs sharing its terminal.
    os.setsid()
    collector = multiprocessing.parent_process().sentinel
    # A forked engine child can inherit the control pipe. Watch the
    # watchdog's parent PID as well so a worker crash still tears down the group.
    with connection, subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", _WATCH_COLLECTOR, str(collector), str(os.getpid())],
        stdin=subprocess.PIPE,
        pass_fds=(collector,),
    ) as watchdog:
        status = b"!"
        try:
            _serve(connection, factory, spec, options)
            status = b"q"
        finally:
            watchdog.stdin.write(status)


def _serve(connection, factory, spec, options):
    try:
        backend = factory(spec, **options)
    except Exception as error:
        connection.send(("fatal", format_error(error)))
        return
    connection.send(("ok", backend.model_type))
    while True:
        try:
            requests = connection.recv()
        except EOFError:
            return
        try:
            results = backend.generate_many(requests)
            reply = ("ok", [(False, format_error(value)) if isinstance(value, Exception)
                             else (True, value) for value in results])
        except BatchEngineUnavailable as error:
            connection.send(("fatal", format_error(error)))
            return
        except Exception as error:
            reply = ("error", format_error(error))
        connection.send(reply)


class VLLMWorker:
    """Supervise the engine while the caller retains ownership of the journal."""

    def __init__(self, spec, *, timeout_seconds=DEFAULT_ENGINE_TIMEOUT, backend_factory=VLLMBackend,
                 **options):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("vLLM engine timeout must be positive and finite")
        self.spec = spec
        self.timeout_seconds = timeout_seconds
        self.backend_factory = backend_factory
        self.options = options

    def __enter__(self):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_run_worker,
                                       args=(child, self.backend_factory, self.spec, self.options))
        self.io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vllm-transport")
        try:
            with child:
                self.process.start()
            self.model_type = self._receive("startup")
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *exc_info):
        self.connection.close()
        if self.process.pid is not None:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self._stop_worker()
            self.process.close()
        self.io.shutdown(wait=False, cancel_futures=True)

    def _stop_worker(self):
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # Startup may have timed out before the child could create its group.
            self.process.kill()
        self.process.join()

    def _round_trip(self, requests):
        if requests is not None:
            self.connection.send(requests)
        return self.connection.recv()

    def _receive(self, phase, requests=None):
        future = self.io.submit(self._round_trip, requests)
        try:
            status, value = future.result(timeout=self.timeout_seconds)
        except TimeoutError as error:
            self._stop_worker()
            raise BatchEngineUnavailable(
                f"vLLM engine {phase} timed out after {self.timeout_seconds:g} seconds; "
                "worker stopped. Resume collection from the journal."
            ) from error
        except (EOFError, OSError) as error:
            self._stop_worker()
            raise BatchEngineUnavailable(f"vLLM engine worker disconnected during {phase}") from error
        if status == "fatal":
            self._stop_worker()
            raise BatchEngineUnavailable(value)
        if status == "error":
            raise RuntimeError(value)
        return value

    def generate_many(self, requests):
        return [value if ok else RuntimeError(value) for ok, value in self._receive("batch", requests)]
