"""AIA configuration with the same provider choices as the welcome dialog.

Cloud providers use their defaults. Ollama offers the shared hardware-aware
setup and keeps its measured model selection. The Claude Code demo provider
remains accessible only through configuration.ini.
"""

import tkinter as tk
import types
import webbrowser
from tkinter import ttk

from thonny import first_run, get_workbench, ui_utils
from thonny.languages import tr
from thonny.config_ui import ConfigurationPage

from thonny.plugins.aia._common import AIACore
from thonny.plugins.aia.prompts import answer_style


def style_names():
    """Answer style id -> name in the current language."""
    return {"socratic": tr("Socratic"), "explanatory": tr("Explanatory"), "direct": tr("Direct")}


def _cloud_note(provider):
    return (
        tr(
            "Answers come from %s. You need an API key: create it at the link below, or get one "
            "from your teacher."
        )
        % first_run.COMPANY[provider]
    )


def _later_note():
    return tr("No provider is chosen: AIA stays switched on, but it does not answer.")


class AIAPanelConfigPage(ConfigurationPage):
    def __init__(self, master):
        super().__init__(master)
        self._core = AIACore()
        self._core.register_providers()
        wb = get_workbench()

        self._models = first_run._default_models()
        self._models[first_run.AI_LOCAL] = (
            wb.get_option("aia.ollama_model") or self._models[first_run.AI_LOCAL]
        )
        self._local_selection = None
        self._key_urls = first_run._key_urls()
        self._labels = first_run._ai_labels(self._models)

        stored = wb.get_option("aia.provider") or first_run.AI_LATER
        self._ai = stored if stored in self._labels else first_run.AI_LATER
        # A provider set by hand in configuration.ini (e.g. claude_code) is not offered here:
        # it stays until the choice is changed on the page
        self._kept_provider = None if stored in self._labels else stored
        self._ollama = None  # last first_run.ollama_state(), checked when the local model is shown
        self._keys = {}  # provider -> key as edited on this page
        self._link_url = ""

        row = 0
        self._enabled_var = wb.get_variable("aia.enabled")
        ttk.Checkbutton(
            self,
            text=tr("Use AI-based AIA instead of the built-in Assistant"),
            variable=self._enabled_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 10))
        row += 1

        ttk.Label(self, text=tr("AI assistant:")).grid(row=row, column=0, sticky="w", pady=2)
        self._ai_var = tk.StringVar(value=self._labels[self._ai])
        self._ai_combo = ttk.Combobox(
            self,
            textvariable=self._ai_var,
            values=list(self._labels.values()),
            state="readonly",
            exportselection=False,
            width=36,
        )
        self._ai_combo.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        self._ai_combo.bind("<<ComboboxSelected>>", self._on_ai_changed)
        row += 1

        self._detail = self._make_detail(ttk.Frame(self))
        self._detail.frame.grid(row=row, column=0, columnspan=2, sticky="nw", pady=(4, 8))
        row += 1

        # --- Answer style ---
        ttk.Label(self, text=tr("Answer style:")).grid(row=row, column=0, sticky="w", pady=2)
        self._mentoring_var = wb.get_variable("aia.mentoring_level")
        self._style_names = style_names()
        self._style_display_var = tk.StringVar(
            value=self._style_names.get(answer_style(self._mentoring_var.get()))
        )
        style_combo = ttk.Combobox(
            self,
            textvariable=self._style_display_var,
            values=list(self._style_names.values()),
            state="readonly",
            width=18,
        )
        style_combo.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        style_combo.bind("<<ComboboxSelected>>", self._on_style_changed)
        row += 1

        self._legend = ttk.Label(
            self,
            text=tr(
                "Socratic = guides you to the answer, Explanatory = explains and answers, "
                "Direct = short answer only"
            ),
            foreground="#7f8c8d",
        )
        self._legend.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 8))
        row += 1

        # --- Context section (separated) ---
        ttk.Separator(self, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(5, 5)
        )
        row += 1

        ttk.Label(self, text=tr("Context:"), font="TkDefaultFont 9 bold").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 2)
        )
        row += 1

        ttk.Checkbutton(
            self,
            text=tr("Automatically include current file as context"),
            variable=wb.get_variable("aia.auto_attach_context"),
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Checkbutton(
            self,
            text=tr("Also include locally imported files"),
            variable=wb.get_variable("aia.include_local_imports"),
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self._fix_detail_size()
        self._show(self._ai)

    # --- AI choice ---

    def _make_detail(self, frame):
        d = types.SimpleNamespace(frame=frame)
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(4, weight=1)
        d.note = ttk.Label(frame, wraplength=460, justify="left")
        d.note.grid(row=1, column=1, sticky="nw")
        d.key_frame = ttk.Frame(frame)
        d.key_frame.grid(row=2, column=1, sticky="nw", pady=(7, 0))
        ttk.Label(d.key_frame, text=tr("API key:")).grid(row=1, column=1, sticky="w")
        self._key_var = tk.StringVar()
        ttk.Entry(d.key_frame, textvariable=self._key_var, show="*", width=36).grid(
            row=1, column=2, sticky="w", padx=(10, 0)
        )
        # Thonny 5 has no secrets store (the API key dialog said so before, keep saying it)
        ttk.Label(
            d.key_frame,
            text=tr("The key is stored in plain text in Thonny's configuration file."),
            foreground="#7f8c8d",
        ).grid(row=2, column=1, columnspan=2, sticky="w", pady=(2, 0))
        d.check_button = ttk.Button(
            frame,
            text=tr("Automatically prepare a model (≥10 tokens/s)"),
            command=self._on_check_ollama,
        )
        d.check_button.grid(row=2, column=1, sticky="nw", pady=(7, 0))
        d.link = ui_utils.create_action_label(frame, "", lambda e: webbrowser.open(self._link_url))
        d.link.grid(row=3, column=1, sticky="nw", pady=(3, 0))
        return d

    def _fill(self, ai, ollama):
        """Show the detail of one choice (and one Ollama state for the local model)."""
        d = self._detail
        link = ""
        if ai in first_run.AI_CLOUD:
            d.note.configure(text=_cloud_note(ai))
            d.key_frame.grid()
            d.check_button.grid_remove()
            link = self._key_urls[ai]
        elif ai == first_run.AI_LOCAL:
            d.note.configure(text=first_run._local_note(ollama, self._models[ai]))
            d.key_frame.grid_remove()
            d.check_button.grid()
            link = first_run.OLLAMA_URL if ollama in ("missing", "stopped") else ""
        else:
            d.note.configure(text=_later_note())
            d.key_frame.grid_remove()
            d.check_button.grid_remove()
        d.link.configure(text=link)
        if link:
            d.link.grid()
        else:
            d.link.grid_remove()
        self._link_url = link

    def _fix_detail_size(self):
        """Size the detail area for its largest variant before the page is shown, so that
        changing the choice moves nothing below it."""
        variants = [(ai, None) for ai in self._labels if ai != first_run.AI_LOCAL]
        variants += [(first_run.AI_LOCAL, s) for s in ("missing", "stopped", "no_model", "ready")]
        # the note is as wide as the page's widest line (the legend of the answer styles)
        self.update_idletasks()
        self._detail.note.configure(
            wraplength=max(self._legend.winfo_reqwidth(), self._ai_combo.winfo_reqwidth())
        )
        width = height = 0
        for ai, ollama in variants:
            self._fill(ai, ollama)
            self._detail.frame.update_idletasks()
            width = max(width, self._detail.frame.winfo_reqwidth())
            height = max(height, self._detail.frame.winfo_reqheight())
        self._detail.frame.configure(width=width, height=height)
        self._detail.frame.grid_propagate(False)

    def _show(self, ai):
        if ai == first_run.AI_LOCAL and self._ollama is None:
            self._ollama = first_run.ollama_state(self._models[ai])
        if ai in first_run.AI_CLOUD:
            if ai not in self._keys:
                self._keys[ai] = self._core._get_stored_api_key(ai) or ""
            self._key_var.set(self._keys[ai])
        self._fill(ai, self._ollama)

    def _on_ai_changed(self, event=None):
        if self._ai in first_run.AI_CLOUD:
            self._keys[self._ai] = self._key_var.get()
        self._ai = list(self._labels)[self._ai_combo.current()]
        self._kept_provider = None
        self._show(self._ai)

    def _on_check_ollama(self):
        from thonny.plugins.aia.model_setup_ui import ModelSetupDialog

        dialog = ModelSetupDialog(self)
        dialog.show_modal()
        if dialog.result is not None:
            self._local_selection = dialog.result
            self._models[first_run.AI_LOCAL] = dialog.result.model
        self._ollama = first_run.ollama_state(self._models[first_run.AI_LOCAL])
        self._fill(first_run.AI_LOCAL, self._ollama)

    def _on_style_changed(self, event=None):
        shown = self._style_display_var.get()
        for style, name in self._style_names.items():
            if name == shown:
                self._mentoring_var.set(style)

    def apply(self, changed_options=None):
        # 4.x/5.0 call apply() without arguments, newer Thonny passes changed_options
        wb = get_workbench()
        if (
            self._ai == first_run.AI_LOCAL
            and self._local_selection is None
            and not wb.get_option("aia.ollama_model")
        ):
            self._on_check_ollama()
            if self._local_selection is None:
                return False
        if self._ai in first_run.AI_CLOUD:
            self._keys[self._ai] = self._key_var.get()
        for provider, key in self._keys.items():
            if key.strip() != (self._core._get_stored_api_key(provider) or ""):
                self._core.set_api_key(provider, key)
        if self._kept_provider is None:
            wb.set_option("aia.provider", "" if self._ai == first_run.AI_LATER else self._ai)
        wb.set_option("aia.model", "")  # a model chosen in earlier versions no longer applies
        if self._ai == first_run.AI_LOCAL and self._local_selection:
            wb.set_option("aia.ollama_model", self._local_selection.model)
            wb.set_option("aia.ollama_benchmark_tps", self._local_selection.tokens_per_second)
            wb.set_option("aia.ollama_host", "http://127.0.0.1:11434")
        return True
