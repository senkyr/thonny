"""Provisioning policy, measured fallback and interruption without network/downloads."""

import json
import threading
from types import SimpleNamespace

import pytest

from thonny.plugins.aia import model_setup as m


@pytest.fixture
def hardware(tmp_path):
    return m.Hardware("Test CPU", 8, 50, str(tmp_path))


class FakeAPI:
    def __init__(self, installed=(), speeds=(20, 20)):
        self.installed = installed
        self.speeds = iter(speeds)
        self.calls = []
        self.cancelled = False

    def get(self, path, timeout=3):
        self.calls.append((path, None))
        return (
            {"models": [{"name": name} for name in self.installed]}
            if path == "tags"
            else {"version": "1"}
        )

    def request(self, path, body, **kwargs):
        self.calls.append((path, body))
        if path == "pull":
            yield {"status": "downloading", "completed": 50, "total": 100}
            yield {"status": "success"}
        elif path == "chat":
            yield {"message": {"content": "Synthetic answer"}}
            yield {"done": True, "eval_count": 128, "eval_duration": 128 / next(self.speeds) * 1e9}

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def make_setup(hardware, monkeypatch):
    def make(api, emit=lambda *args: None):
        setup = m.ModelSetup(hardware=hardware, api=api, emit=emit)
        setup.unloaded = []
        monkeypatch.setattr(setup, "unload", setup.unloaded.append)
        return setup

    return make


@pytest.mark.parametrize(
    "ram,expected",
    [
        (2, []),
        (4, []),
        (8, ["qwen3:4b", "qwen3:1.7b"]),
        (16, ["qwen3:8b", "qwen3:4b", "qwen3:1.7b"]),
        (24, [c.name for c in m.CANDIDATES]),
    ],
)
def test_ram_reserve_filters_candidates(ram, expected):
    assert [c.name for c in m.candidates_for(m.Hardware("CPU", ram, 100, "/tmp"))] == expected


def test_download_measure_and_leave_successful_model_loaded(make_setup):
    api = FakeAPI(speeds=(19, 12))
    setup = make_setup(api)
    result = setup.run()
    assert result.model == "qwen3:4b"
    assert result.tokens_per_second == 12
    assert result.samples == (19, 12)
    assert not setup.unloaded
    assert [path for path, _ in api.calls] == ["version", "tags", "pull", "chat", "chat"]
    for path, body in api.calls:
        if path == "chat":
            assert body["think"] is False
            assert body["options"]["num_ctx"] == m.LOCAL_CONTEXT
            assert body["options"]["num_predict"] == 256


def test_slower_of_two_tests_controls_fallback(make_setup):
    api = FakeAPI(installed=("qwen3:4b", "qwen3:1.7b"), speeds=(30, 9.9, 10, 11))
    setup = make_setup(api)
    result = setup.run()
    assert result.model == "qwen3:1.7b"
    assert result.tokens_per_second == 10
    assert setup.unloaded == ["qwen3:4b"]
    assert all(path != "pull" for path, _ in api.calls)


def test_no_model_meets_threshold(make_setup):
    setup = make_setup(FakeAPI(speeds=(8, 9, 9, 9)))
    with pytest.raises(RuntimeError, match="No tested model"):
        setup.run()
    assert setup.unloaded == ["qwen3:4b", "qwen3:1.7b"]


def test_cancel_during_download_prevents_benchmark(make_setup):
    api = FakeAPI()
    setup = make_setup(api)

    def emit(kind, *args):
        if kind == "progress":
            setup.cancel()

    setup.emit = emit
    with pytest.raises(m.SetupCancelled):
        setup.run()
    assert api.cancelled
    assert not any(path == "chat" for path, _ in api.calls)


def test_cancel_during_benchmark_unloads_model(make_setup):
    setup = make_setup(FakeAPI(installed=("qwen3:4b",)))

    def emit(kind, *args):
        if kind == "status" and args[0].startswith("Measuring"):
            setup.cancel()

    setup.emit = emit
    with pytest.raises(m.SetupCancelled):
        setup.run()
    assert setup.unloaded == ["qwen3:4b"]


def test_disk_full_never_downloads(make_setup, monkeypatch):
    monkeypatch.setattr(m.shutil, "disk_usage", lambda path: SimpleNamespace(free=m.GIB))
    api = FakeAPI()
    setup = make_setup(api)
    with pytest.raises(RuntimeError, match="No tested model"):
        setup.run()
    assert not any(path == "pull" for path, _ in api.calls)
    assert all("disk space" in attempt["error"] for attempt in setup.attempts)


def test_disk_full_can_still_use_installed_model(make_setup, monkeypatch):
    monkeypatch.setattr(m.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    setup = make_setup(FakeAPI(installed=("qwen3:4b",)))
    assert setup.run().model == "qwen3:4b"


def test_incomplete_pull_is_not_used(make_setup):
    api = FakeAPI()
    original = api.request

    def request(path, body, **kwargs):
        if path == "pull":
            yield {"status": "downloading"}
        else:
            yield from original(path, body, **kwargs)

    api.request = request
    setup = make_setup(api)
    with pytest.raises(RuntimeError, match="No tested model"):
        setup.run()
    assert not any(path == "chat" for path, _ in api.calls)


@pytest.mark.parametrize(
    "changes",
    [
        {"done": False},
        {"eval_count": 3},
        {"eval_duration": 0},
        {"message": {"content": ""}},
    ],
)
def test_invalid_measurement_never_qualifies(changes):
    response = {
        "done": True,
        "eval_count": 128,
        "eval_duration": 1e9,
        "message": {"content": "answer"},
    }
    response.update(changes)
    with pytest.raises(ValueError):
        m.measured_tps(response)


def test_windows_cli_detection_without_path(monkeypatch, tmp_path):
    executable = tmp_path / "Programs/Ollama/ollama.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fixture")
    monkeypatch.setattr(m.platform, "system", lambda: "Windows")
    monkeypatch.setattr(m.shutil, "which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(m.os, "access", lambda *args: True)
    assert m.find_ollama() == str(executable)


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_server_start_is_local_and_has_no_shell(system, monkeypatch, hardware):
    api = FakeAPI()
    count = []

    def get(path, timeout=3):
        count.append(path)
        if len(count) == 1:
            raise ConnectionRefusedError()
        return {"version": "1"}

    api.get = get
    monkeypatch.setattr(m.platform, "system", lambda: system)
    monkeypatch.setattr(m, "find_ollama", lambda: "/path with spaces/ollama")
    monkeypatch.setattr(m.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "https://remote.example")
    starts = []
    monkeypatch.setattr(m.subprocess, "Popen", lambda args, **kwargs: starts.append((args, kwargs)))
    m.ModelSetup(hardware=hardware, api=api).ensure_server()
    args, kwargs = starts[0]
    assert args == ["/path with spaces/ollama", "serve"]
    assert kwargs["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert not kwargs.get("shell")
    if system == "Windows":
        assert kwargs["creationflags"] == 0x08000000
    else:
        assert kwargs["start_new_session"] is True


def test_missing_ollama_reports_install_link(monkeypatch):
    api = FakeAPI()
    api.get = lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionRefusedError())
    monkeypatch.setattr(m, "find_ollama", lambda: None)
    with pytest.raises(RuntimeError, match="https://ollama.com/download"):
        m.ModelSetup(api=api).ensure_server()


def test_saved_model_is_used_and_frozen_before_worker(fake_thonny):
    from thonny.plugins.aia.providers.ollama_provider import OllamaProvider

    fake_thonny.options["aia.ollama_model"] = "qwen3:4b"
    provider = OllamaProvider(get_model=lambda: fake_thonny.get_option("aia.ollama_model"))
    frozen = provider.frozen()
    fake_thonny.options["aia.ollama_model"] = "qwen3:1.7b"
    assert frozen.get_default_model() == "qwen3:4b"
    assert provider.get_default_model() == "qwen3:1.7b"


def test_http_api_is_fixed_to_loopback_and_surfaces_stream_errors(monkeypatch):
    calls = []

    class Connection:
        sock = None

        def __init__(self, host, port, **kwargs):
            calls.append((host, port))

        def request(self, method, path, body, headers):
            calls.append((method, path, json.loads(body)))

        def getresponse(self):
            class Response:
                status = 200

                def close(self):
                    pass

                def __iter__(self):
                    yield b'{"error":"download failed"}\n'

            return Response()

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(m.http.client, "HTTPConnection", Connection)
    monkeypatch.setenv("OLLAMA_HOST", "https://remote.example")
    with pytest.raises(RuntimeError, match="download failed"):
        list(m.LocalAPI(threading.Event()).request("pull", {"model": "x"}, stream=True))
    assert calls[0] == ("127.0.0.1", 11434)
    assert calls[-1] == "closed"


def test_first_run_setup_saves_model_without_workbench(monkeypatch, hardware):
    from thonny import first_run
    from thonny.plugins.aia import model_setup_ui

    selection = m.Selection("qwen3:4b", 12.0, (14, 12), hardware)
    shown = []

    class Dialog:
        result = selection

        def __init__(self, parent):
            pass

        def show_modal(self):
            shown.append(True)

    monkeypatch.setattr(model_setup_ui, "ModelSetupDialog", Dialog)
    monkeypatch.setattr(first_run, "ollama_state", lambda model: "ready")
    window = first_run.FirstRunWindow.__new__(first_run.FirstRunWindow)
    window.models = first_run._default_models()
    window._local_selection = None
    window._show_detail = lambda *args: None
    window.detail = None
    window.ai = first_run.AI_LOCAL
    window.mode = first_run.STD_MODE
    window.language_variable = SimpleNamespace(get=lambda: "Čeština")
    window.key_variable = SimpleNamespace(get=lambda: "")
    window.destroy = lambda: None
    saved = {}
    window.conf = SimpleNamespace(set_option=saved.__setitem__, save=lambda: None)
    window.on_ok()
    assert shown == [True]
    assert saved["aia.ollama_model"] == "qwen3:4b"
    assert saved["aia.ollama_benchmark_tps"] == 12.0
    assert saved["aia.ollama_host"] == "http://127.0.0.1:11434"
    assert window.ok


@pytest.mark.parametrize("framing", ["eof", "length", "chunked"])
def test_api_cancellation_interrupts_a_stalled_stream(monkeypatch, framing):
    import http.server

    ready = threading.Event()
    reader_ready = threading.Event()
    release = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            if framing == "length":
                self.send_header("Content-Length", "100")
            elif framing == "chunked":
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.flush()
            ready.set()
            release.wait(5)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    connection = m.http.client.HTTPConnection

    class TestConnection(connection):
        def getresponse(self):
            response = super().getresponse()
            reader_ready.set()
            return response

    monkeypatch.setattr(
        m.http.client,
        "HTTPConnection",
        lambda host, port, **kw: TestConnection(host, server.server_port, **kw),
    )
    api = m.LocalAPI(threading.Event())
    outcomes = []
    # Windows may leave an already-blocked recv waiting after shutdown. Verify
    # cancellation independently of that OS-specific interruption mechanism.
    monkeypatch.setattr(m.socket.socket, "shutdown", lambda *args: None)

    def read():
        try:
            list(api.request("pull", {"model": "fixture"}, stream=True))
        except m.SetupCancelled:
            outcomes.append("cancelled")

    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    try:
        assert ready.wait(2)
        assert reader_ready.wait(2)
        api.cancel()
        worker.join(2)
        assert not worker.is_alive()
        assert outcomes == ["cancelled"]
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def test_cancellable_reader_preserves_inactivity_timeout():
    left, right = m.socket.socketpair()
    left.settimeout(0.1)
    reader = m._CancellableReader(left, threading.Event())
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            reader.readinto(bytearray(16))
    finally:
        reader.close()
        left.close()
        right.close()
