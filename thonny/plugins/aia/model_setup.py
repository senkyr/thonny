"""Local-only Ollama provisioning and measured model selection (no Tk dependency).

Memory estimates only shortlist candidates. A model qualifies only after two
representative chat completions each reach the requested generation throughput.
The measurement is not a guarantee under future load or for longer conversations.
"""

import ctypes
from dataclasses import dataclass
import http.client
import io
import json
import os
from pathlib import Path
import platform
import re
import select
import shutil
import socket
import subprocess
import threading
import time

from thonny.plugins.aia.providers.ollama_provider import KEEP_ALIVE, LOCAL_CONTEXT, think_setting

GIB = 1024**3
TARGET_TPS = 10.0


@dataclass(frozen=True)
class Candidate:
    name: str
    ram_gb: float
    download_gb: float


# Preference order, not a claim that parameter count alone measures teaching quality.
CANDIDATES = (
    Candidate("gpt-oss:20b", 16, 13),
    Candidate("qwen3:8b", 7, 5.2),
    Candidate("qwen3:4b", 4, 2.5),
    Candidate("qwen3:1.7b", 2.5, 1.4),
)


@dataclass(frozen=True)
class Hardware:
    cpu: str
    ram_gb: float
    disk_gb: float
    model_dir: str


@dataclass(frozen=True)
class Selection:
    model: str
    tokens_per_second: float
    samples: tuple
    hardware: Hardware


class SetupCancelled(Exception):
    pass


def detect_hardware():
    system = platform.system()
    cpu = platform.processor() or platform.machine()
    if system == "Darwin":

        def sysctl(key):
            return subprocess.check_output(["/usr/sbin/sysctl", "-n", key], text=True).strip()

        total = int(sysctl("hw.memsize"))
        cpu = sysctl("machdep.cpu.brand_string")
    elif system == "Windows":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended",
                )
            ]

        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError("Cannot read system RAM")
        total = memory.total
    else:
        text = Path("/proc/meminfo").read_text()
        total = int(re.search(r"MemTotal:\s+(\d+)", text).group(1)) * 1024
    model_dir = Path(os.environ.get("OLLAMA_MODELS", "~/.ollama/models")).expanduser()
    existing = model_dir
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return Hardware(cpu, total / GIB, shutil.disk_usage(existing).free / GIB, str(model_dir))


def candidates_for(hardware):
    # Leave room for macOS, the editor and the student's program, including KV cache
    # in each candidate's RAM estimate. Actual memory pressure is checked by the run.
    budget = hardware.ram_gb - max(2.0, hardware.ram_gb * 0.25)
    return [candidate for candidate in CANDIDATES if candidate.ram_gb <= budget]


def find_ollama():
    paths = [shutil.which("ollama")]
    if platform.system() == "Darwin":
        paths += [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",
            str(Path.home() / "Applications/Ollama.app/Contents/Resources/ollama"),
        ]
    elif platform.system() == "Windows":
        paths += [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
        ]
    return next((str(p) for p in paths if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)


class _CancellableReader(io.RawIOBase):
    """Poll readiness so cancellation does not depend on cross-thread shutdown."""

    def __init__(self, sock, cancel_event):
        self.sock = sock.dup()
        self.cancel_event = cancel_event

    def readable(self):
        return True

    def readinto(self, buffer):
        timeout = self.sock.gettimeout()
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.cancel_event.is_set():
                raise SetupCancelled()
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("Ollama response timed out")
            ready, _, _ = select.select(
                [self.sock], [], [], min(0.1, remaining) if remaining is not None else 0.1
            )
            if ready:
                if self.cancel_event.is_set():
                    raise SetupCancelled()
                return self.sock.recv_into(buffer)

    def close(self):
        self.sock.close()
        super().close()


class _ResponseSocket:
    def __init__(self, sock, cancel_event):
        self.sock = sock
        self.cancel_event = cancel_event

    def makefile(self, mode):
        return io.BufferedReader(_CancellableReader(self.sock, self.cancel_event))


class LocalAPI:
    """Fixed loopback endpoint: never provision a remote OLLAMA_HOST by accident."""

    def __init__(self, cancel_event):
        self.cancel_event = cancel_event
        self._connection = None
        self._lock = threading.Lock()

    def cancel(self):
        self.cancel_event.set()
        with self._lock:
            conn = self._connection
            sock = conn.sock if conn else None
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def request(self, path, body=None, stream=False, timeout=90):
        if self.cancel_event.is_set():
            raise SetupCancelled()
        conn = http.client.HTTPConnection("127.0.0.1", 11434, timeout=timeout)
        conn.response_class = lambda sock, *args, **kwargs: http.client.HTTPResponse(
            _ResponseSocket(sock, self.cancel_event), *args, **kwargs
        )
        with self._lock:
            self._connection = conn
        response = None
        try:
            conn.request(
                "POST" if body is not None else "GET",
                "/api/" + path,
                json.dumps(body) if body is not None else None,
                {"Content-Type": "application/json"},
            )
            if self.cancel_event.is_set():
                raise SetupCancelled()
            response = conn.getresponse()
            if response.status != 200:
                error = response.read(4096).decode("utf-8", errors="replace")
                raise RuntimeError(f"Ollama HTTP {response.status}: {error}")
            if stream:
                for line in response:
                    if self.cancel_event.is_set():
                        raise SetupCancelled()
                    if line.strip():
                        item = json.loads(line)
                        if item.get("error"):
                            raise RuntimeError(item["error"])
                        yield item
                if self.cancel_event.is_set():
                    raise SetupCancelled()
            else:
                item = json.load(response)
                if item.get("error"):
                    raise RuntimeError(item["error"])
                yield item
        except Exception:
            if self.cancel_event.is_set():
                raise SetupCancelled() from None
            raise
        finally:
            if response is not None:
                response.close()
            conn.close()
            with self._lock:
                if self._connection is conn:
                    self._connection = None

    def get(self, path, timeout=3):
        return next(self.request(path, timeout=timeout))

    def post(self, path, body, timeout=90):
        return next(self.request(path, body, timeout=timeout))


BENCHMARK_PROMPTS = (
    "Vysvětli začátečníkovi česky, proč range(5) neobsahuje číslo 5. "
    "Napiš souvislé vysvětlení s příkladem a otázkou na ověření pochopení, alespoň 150 slov.",
    "Vysvětli začátečníkovi česky chybu v Pythonu: cisla = [1, 2, 3]; print(cisla[3]). "
    "Napiš souvislé vysvětlení indexování s příkladem a otázkou, alespoň 150 slov.",
)


def measured_tps(response):
    tokens = response.get("eval_count", 0)
    duration = response.get("eval_duration", 0)
    if not response.get("done") or tokens < 64 or duration <= 0:
        raise ValueError("Incomplete or too short benchmark")
    if not response.get("message", {}).get("content", "").strip():
        raise ValueError("Benchmark produced no answer")
    return tokens / duration * 1e9


class ModelSetup:
    def __init__(self, emit=lambda *args: None, hardware=None, api=None, cancel_event=None):
        self.emit = emit
        self.cancel_event = cancel_event or threading.Event()
        self.api = api or LocalAPI(self.cancel_event)
        self.hardware = hardware
        self.attempts = []
        self._active_model = None

    def cancel(self):
        self.cancel_event.set()
        self.api.cancel()

    def check_cancelled(self):
        if self.cancel_event.is_set():
            raise SetupCancelled()

    def ensure_server(self):
        try:
            self.api.get("version")
            return
        except SetupCancelled:
            raise
        except Exception:
            pass
        self.emit("status", "Starting Ollama…", ())
        executable = find_ollama()
        if not executable:
            raise RuntimeError(
                "Ollama is not installed. Install it from https://ollama.com/download and try again."
            )
        env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434")
        kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "env": env}
        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        # Leave the server running for AIA's keeper and later sessions.
        process = subprocess.Popen([executable, "serve"], **kwargs)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            self.check_cancelled()
            try:
                self.api.get("version", timeout=1)
                return
            except SetupCancelled:
                raise
            except Exception:
                if process.poll() is not None:
                    break
                self.cancel_event.wait(0.25)
        raise RuntimeError("Ollama could not start. Start Ollama and try again.")

    def unload(self, model):
        # Fresh connection also works after cancellation interrupted the active request.
        try:
            LocalAPI(threading.Event()).post(
                "generate", {"model": model, "keep_alive": 0}, timeout=3
            )
        except Exception:
            pass

    def benchmark(self, model):
        from thonny.plugins.aia.prompts import SOCRATIC_PROMPT

        samples = []
        for index, prompt in enumerate(BENCHMARK_PROMPTS, 1):
            self.emit("status", "Measuring %s (%s/2)…", (model, index))
            last = {}
            for item in self.api.request(
                "chat",
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SOCRATIC_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "think": think_setting(model),
                    "stream": True,
                    "keep_alive": KEEP_ALIVE,
                    "options": {"num_ctx": LOCAL_CONTEXT, "num_predict": 256, "temperature": 0},
                },
                stream=True,
            ):
                self.check_cancelled()
                # Ollama puts the counters in the final chunk and content in earlier chunks.
                content = last.get("message", {}).get("content", "") + item.get("message", {}).get(
                    "content", ""
                )
                last = dict(item, message={"content": content})
            samples.append(measured_tps(last))
        return tuple(samples)

    def run(self):
        try:
            hardware = self.hardware or detect_hardware()
            self.emit("hardware", hardware, ())
            candidates = candidates_for(hardware)
            if not candidates:
                raise RuntimeError(
                    "Not enough RAM for the supported local models. Use a cloud provider."
                )
            self.ensure_server()
            installed = {
                m.get("name") or m.get("model") for m in self.api.get("tags").get("models", [])
            }
            for candidate in candidates:
                self.check_cancelled()
                model = candidate.name
                self.emit("candidate", candidate, ())
                try:
                    if model not in installed:
                        existing = Path(hardware.model_dir)
                        while not existing.exists() and existing != existing.parent:
                            existing = existing.parent
                        if shutil.disk_usage(existing).free / GIB < candidate.download_gb + 1:
                            raise RuntimeError("Not enough disk space to download the model.")
                        self.emit(
                            "status",
                            "Downloading %s (about %s GB)…",
                            (model, candidate.download_gb),
                        )
                        complete = False
                        last_progress = 0.0
                        for chunk in self.api.request(
                            "pull", {"model": model, "stream": True}, stream=True
                        ):
                            self.check_cancelled()
                            now = time.monotonic()
                            if now - last_progress >= 0.1 or chunk.get("status") == "success":
                                self.emit("progress", chunk, ())
                                last_progress = now
                            complete = chunk.get("status") == "success"
                        if not complete:
                            raise RuntimeError("Model download did not finish.")
                    self._active_model = model
                    samples = self.benchmark(model)
                    speed = min(samples)
                    self.attempts.append({"model": model, "samples": samples, "tps": speed})
                    self.emit("measured", model, samples)
                    if speed >= TARGET_TPS:
                        self.check_cancelled()
                        return Selection(model, speed, samples, hardware)
                except SetupCancelled:
                    raise
                except Exception as exc:
                    self.attempts.append({"model": model, "error": str(exc)})
                    self.emit("failed", model, (str(exc),))
                self.unload(model)
                self._active_model = None
            raise RuntimeError(
                "No tested model reached 10 tokens/s. Close other apps and try again, or use a cloud provider."
            )
        except BaseException:
            if self._active_model:
                self.unload(self._active_model)
                self._active_model = None
            raise
