import logging
from typing import Dict, Iterator, List, Optional, Union
from urllib.parse import urlparse

from thonny.plugins.aia.providers.base import REQUEST_TIMEOUT, LLMProvider

logger = logging.getLogger(__name__)

# Model listing runs on the UI thread (config page) - keep it short.
LIST_TIMEOUT = 3.0

# A local model on CPU may send nothing for a long time: loading gpt-oss:20b took 35-55 s and
# processing the prompt up to 40 s more (2026-10-10). REQUEST_TIMEOUT (60 s) cut it off.
LOCAL_TIMEOUT = 300.0

# How long Ollama keeps the model loaded after a request. Thonny AIA renews it while it
# runs (thonny.plugins.aia.local_model), so the model stays loaded for the whole lesson.
KEEP_ALIVE = "10m"
LOCAL_CONTEXT = 4096

# Loading gpt-oss:20b from disk took 35-55 s on CPU (2026-10-10)
LOAD_TIMEOUT = 300.0

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def is_local_host(host: Optional[str]) -> bool:
    """True for the default server and for addresses on this computer."""
    if not host:
        return True  # OLLAMA_HOST or http://localhost:11434
    name = urlparse(host if "://" in host else f"http://{host}").hostname or ""
    return name in LOCAL_HOSTS or name.startswith("127.")


def _full_name(model: str) -> str:
    """Ollama lists "gemma3" as "gemma3:latest"."""
    return model if ":" in model else model + ":latest"


def think_setting(model: str) -> Union[bool, str]:
    """Least thinking the model allows, because the student sees nothing while it thinks.

    gpt-oss cannot switch thinking off - it ignores think=False and thinks at its
    default level; only "low"/"medium"/"high" work (checked 2026-10-09: with False
    200 tokens of thinking and no answer, with "low" a one-word answer). Other
    thinking models (Gemma 4, Qwen 3 ...) take False; models without thinking accept it.
    """
    return "low" if model.split(":")[0].split("/")[-1].startswith("gpt-oss") else False


class OllamaProvider(LLMProvider):
    """Provider for locally running Ollama models."""

    def __init__(self, get_host=None, get_model=None):
        self._get_host = get_host
        self._get_model = get_model

    def _client(self, timeout: float):
        import ollama

        host = self._get_host() if self._get_host else None
        kwargs = {}
        if is_local_host(host):
            # httpx takes proxies from the Windows registry and has no bypass for local
            # addresses there: a school proxy answered requests to 127.0.0.1 with 503
            # (PC-S210C, 2026-10-10). A local server never needs a proxy.
            kwargs["trust_env"] = False
        # host=None -> OLLAMA_HOST env var or http://localhost:11434
        return ollama.Client(host=host or None, timeout=timeout, **kwargs)

    def stream_chat(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        model: str,
    ) -> Iterator[str]:
        api_messages = [{"role": "system", "content": system_prompt}] + list(messages)
        stream = self._client(max(REQUEST_TIMEOUT, LOCAL_TIMEOUT)).chat(
            model=model,
            messages=api_messages,
            stream=True,
            think=think_setting(model),
            keep_alive=KEEP_ALIVE,
            options={"num_ctx": LOCAL_CONTEXT},
        )
        try:
            for chunk in stream:
                text = chunk["message"]["content"]
                if text:
                    yield text
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

    def preload(self, model: str) -> None:
        """Load the model into memory (or renew its keep_alive). Blocks - worker thread only."""
        host = self._get_host() if self._get_host else None
        if self._get_model and self._get_model() and host == "http://127.0.0.1:11434":
            from thonny.plugins.aia.model_setup import ModelSetup

            ModelSetup().ensure_server()
        # A request without a prompt only loads the model (Ollama API docs, "Load a model")
        self._client(LOAD_TIMEOUT).generate(
            model=model, keep_alive=KEEP_ALIVE, options={"num_ctx": LOCAL_CONTEXT}
        )

    def is_loaded(self, model: str, timeout: float = LIST_TIMEOUT) -> bool:
        """Whether the model is in memory (/api/ps). Blocks - worker thread only."""
        wanted = _full_name(model)
        response = self._client(timeout).ps()
        return any(
            _full_name(m.get("model") or m.get("name") or "") == wanted
            for m in response.get("models", [])
        )

    def unload(self, model: str, timeout: float = LIST_TIMEOUT) -> None:
        """Free the model's memory now. Blocks."""
        self._client(timeout).generate(model=model, keep_alive=0)

    def is_available(self) -> bool:
        # No key needed. Whether the server runs is found out by the request
        # itself (friendly "cannot reach Ollama" error) - no blocking ping here.
        return True

    def get_display_name(self) -> str:
        return "Ollama (local)"

    def get_default_model(self) -> str:
        if self._get_model:
            selected = self._get_model()
            if selected:
                return selected
        # gpt-oss:20b (13 GB, needs ~14 GB free RAM): the only small-enough local model
        # that gave correct code in all 6 test answers and readable Czech (2026-10-09/10).
        # Speed is a matter of the user's hardware: on CPU only (i7-8550U, Ryzen 5 1600)
        # it generates 4.5-5 tokens/s, i.e. 20-70 s per answer. Gemma 3 4B, Gemma 4 E2B
        # and Nemotron 3 Nano 4B were faster but wrong in a third of the answers.
        return "gpt-oss:20b"

    def get_available_models(self) -> List[str]:
        """Installed models, or the default when the server is unreachable."""
        try:
            response = self._client(LIST_TIMEOUT).list()
            models = [m.get("name", m.get("model", "")) for m in response.get("models", [])]
            models = [m for m in models if m]
            return models if models else [self.get_default_model()]
        except Exception:
            logger.debug("Could not list Ollama models", exc_info=True)
            return [self.get_default_model()]
