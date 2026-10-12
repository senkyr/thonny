"""Keep the local model (Ollama) loaded while Thonny runs.

Loading gpt-oss:20b takes 35-55 s on CPU and Ollama unloads an idle model after
5 minutes, so the first error explanation of a lesson would wait almost a minute
just for the load. Instead the model is loaded at startup and its keep_alive is
renewed periodically; every chat request renews it too (``KEEP_ALIVE`` in the
provider). A refresh instead of ``keep_alive=-1`` means a crashed Thonny frees
the memory (about 13 GB) within ``KEEP_ALIVE`` - a clean exit frees it at once.

A failed load (Ollama not started yet - its tray app starts with Windows and may come
up after Thonny - or the model not pulled) is retried after ``RETRY_MS``, not after
``REFRESH_MS``: otherwise the first error of a lesson waited for the whole load anyway.

Threading: the timer and option reads run on the UI thread, the HTTP requests in
a daemon thread with a frozen provider.
"""

import logging
import threading

from thonny.plugins.aia.providers.ollama_provider import KEEP_ALIVE

logger = logging.getLogger(__name__)

# Renew well before KEEP_ALIVE (10 min) runs out
REFRESH_MS = 4 * 60 * 1000
FIRST_LOAD_DELAY_MS = 1000
RETRY_MS = 15 * 1000


class LocalModelKeeper:
    def __init__(self, wb, core):
        self._wb = wb
        self._core = core
        self._loaded = None  # (frozen provider, model) that we asked Ollama to keep
        self._timer = None
        # State of the last load, written by its worker thread, read by the timer
        self._loading = False
        self._failed = False

    def install(self):
        try:
            self._wb.get_variable("aia.provider").trace_add("write", self._on_settings_changed)
            self._wb.get_variable("aia.enabled").trace_add("write", self._on_settings_changed)
            self._wb.get_variable("aia.ollama_model").trace_add("write", self._on_settings_changed)
        except Exception:
            logger.debug("Could not trace AIA options", exc_info=True)
        self._wb.bind("WorkbenchClose", self._on_close, True)
        self._schedule(FIRST_LOAD_DELAY_MS)

    def wanted(self):
        """(frozen provider, model) to keep loaded, or None. UI thread only."""
        wb = self._wb
        if not wb.get_option("aia.enabled") or wb.get_option("aia.provider") != "ollama":
            return None
        provider = self._core.providers.get("ollama")
        if provider is None:
            return None
        return provider.frozen(), self._core.get_current_model(provider)

    def _schedule(self, delay_ms, func=None):
        if self._timer is not None:
            try:
                self._wb.after_cancel(self._timer)
            except Exception:
                pass
        self._timer = self._wb.after(delay_ms, func or self._tick)

    def _on_settings_changed(self, *_args):
        # The options may change several at once (Options dialog OK) - react once
        self._schedule(FIRST_LOAD_DELAY_MS)

    def _tick(self):
        self._timer = None
        wanted = self.wanted()
        previous = self._loaded
        if previous is not None and (wanted is None or wanted[1] != previous[1]):
            self._run(previous[0].unload, previous[1])
        self._loaded = wanted
        if wanted is not None:
            self._run(wanted[0].preload, wanted[1], load=True)
            self._schedule(RETRY_MS, self._check)

    def _check(self):
        """RETRY_MS after a load started: retry a failed load, else wait for the refresh."""
        self._timer = None
        if self._loading:
            self._schedule(RETRY_MS, self._check)  # loading from disk takes ~30 s
        elif self._failed:
            self._tick()
        else:
            self._schedule(REFRESH_MS)

    def _run(self, func, model, load=False):
        if load:
            self._loading = True
            self._failed = False

        def work():
            try:
                func(model)
            except Exception:
                # Server not running or model not pulled - the chat request reports it
                logger.info("AIA: local model %s: %s failed", model, func.__name__, exc_info=True)
                if load:
                    self._failed = True
            finally:
                if load:
                    self._loading = False

        threading.Thread(target=work, daemon=True).start()

    def _on_close(self, event=None):
        if self._loaded is None:
            return
        provider, model = self._loaded
        self._loaded = None
        try:
            provider.unload(model, timeout=2.0)  # synchronous: the process ends right after
        except Exception:
            logger.debug("Could not unload the local model", exc_info=True)
