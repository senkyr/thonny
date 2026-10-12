# Automatic local model setup (macOS and Windows)

The welcome dialog and Tools → Options → AI Assistant offer **Automatically
prepare a model (≥10 tokens/s)**. On first run, choosing local AI and continuing
also opens this setup. The user can cancel and choose a cloud provider or set up
AI later. Ollama must be installed separately from https://ollama.com/download.

Setup detects the CPU, total RAM and free disk space. Ollama chooses CPU/GPU
acceleration itself. RAM estimates filter a curated preference list, reserving
the larger of 2 GiB and 25% of system memory for the OS and other applications:

| Model | Estimated RAM including context | Approximate download |
| --- | ---: | ---: |
| gpt-oss:20b | 16 GiB | 13 GB |
| qwen3:8b | 7 GiB | 5.2 GB |
| qwen3:4b | 4 GiB | 2.5 GB |
| qwen3:1.7b | 2.5 GiB | 1.4 GB |

These are conservative estimates, not guarantees against memory pressure or
swap. Model sizes can change. The order expresses a preference for the original
AIA model followed by smaller multilingual models; it does not establish equal
teaching quality. Small models still need educational/Czech answer evaluation.

Setup starts an installed Ollama server if necessary, reuses an installed
candidate or streams `/api/pull` with download progress. It tries candidates in
preference order and stops after the first qualifying model. Downloads happen
only after the user starts setup, never just by opening Options. Failed/slow
candidates stay on disk for possible reuse; setup does not delete users' models.
Disk checks use `OLLAMA_MODELS` or `~/.ollama/models`; a separately configured
server storage directory may differ and Ollama then reports a download failure.

Each candidate receives two synthetic Czech Python teaching prompts with the
AIA Socratic system prompt, 4096 context tokens, up to 256 generated tokens, and
the same thinking setting as the assistant. The slower of the two generation
rates must be ≥10 tokens/s. Runs with fewer than 64 tokens, empty answers, errors
or incomplete downloads cannot qualify. Speed is `eval_count / eval_duration *
1e9`, using Ollama's final streaming response. Loading and prompt processing time
are excluded, so this threshold is not time to first token or overall response
latency. For gpt-oss the counters include reasoning tokens as well as visible
output. Speed may decrease with longer conversations, thermal throttling or
other applications; no future minimum is promised.

A successful model is offered for use, kept in memory, and saved when the
welcome dialog or Options is accepted (`aia.ollama_model` and
`aia.ollama_benchmark_tps`). Other providers continue to use their defaults.
The local keeper starts Ollama if needed and loads the selected model on the
next app launch; switching models unloads the old one. A server started by Thonny
is left running for subsequent sessions. Cancelling setup interrupts the active
HTTP connection and unloads the candidate being tested. Downloaded layers may
remain for resumable downloads. If no candidate qualifies, setup reports failure
and does not replace an existing selection.

Provisioning always targets `127.0.0.1:11434`, bypasses proxies, and ignores
`OLLAMA_HOST`. It never downloads or benchmarks on a remote server. Accepting a
prepared model explicitly saves that loopback host for the assistant too.

References: [Ollama chat API](https://docs.ollama.com/api/chat),
[model pull API](https://docs.ollama.com/api/pull),
[Qwen3 model sizes](https://ollama.com/library/qwen3).

Run the offline regression suite with:

```sh
python -B -m pytest thonny/test/aia thonny/test/test_macos_bundle.py -q
```

GitHub Actions runs the tests on macOS and Windows (Python 3.11 and 3.14), and
builds an ad-hoc signed Apple Silicon DMG. GPU throughput is measured locally,
not asserted on a hosted CI runner. See `packaging/mac/README-AIA.md` for the
macOS build and signing limitations.
