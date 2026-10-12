"""Shared, cancellable model setup dialog for first run and Tools > Options."""

import queue
import threading
import tkinter as tk
from tkinter import ttk

from thonny.languages import tr
from thonny.plugins.aia.model_setup import ModelSetup, SetupCancelled


class ModelSetupDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title(tr("Prepare a local AI model"))
        self.transient(master.winfo_toplevel())
        self.result = None
        self._selection = None
        self._prepared = None
        self._closed = False
        self._events = queue.Queue()
        self._setup = ModelSetup(emit=lambda *event: self._events.put(event))
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        frame = ttk.Frame(self, padding=18)
        frame.grid(sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self._hardware = ttk.Label(frame, wraplength=560)
        self._hardware.grid(sticky="w")
        ttk.Label(
            frame,
            wraplength=560,
            text=tr(
                "Downloads a model suited to your RAM and tests it twice. "
                "The target is at least 10 tokens/s in both tests. Downloads may use several GB. "
                "Speed can change with system load; smaller models may give less accurate answers."
            ),
        ).grid(sticky="w", pady=(8, 12))
        self._status = ttk.Label(frame, text=tr("Checking hardware…"), wraplength=560)
        self._status.grid(sticky="w")
        self._progress = ttk.Progressbar(frame, mode="indeterminate", length=540)
        self._progress.grid(sticky="ew", pady=8)
        self._progress.start()
        self._log = tk.Text(frame, width=68, height=8, wrap="word", state="disabled")
        self._log.grid(sticky="nsew")
        frame.rowconfigure(4, weight=1)
        buttons = ttk.Frame(frame)
        buttons.grid(sticky="e", pady=(12, 0))
        self._use = ttk.Button(
            buttons, text=tr("Use this model"), state="disabled", command=self._accept
        )
        self._use.grid(row=0, column=0, padx=8)
        self._cancel_button = ttk.Button(buttons, text=tr("Cancel"), command=self._cancel)
        self._cancel_button.grid(row=0, column=1)
        self.bind("<Escape>", lambda event: self._cancel())
        self.after(100, self._poll)
        threading.Thread(target=self._work, daemon=True).start()

    def show_modal(self):
        # First run has no Workbench yet, so ui_utils.show_dialog cannot be used.
        parent = self.master.winfo_toplevel()
        previous_grab = self.grab_current()
        self.update_idletasks()
        x = max(0, parent.winfo_rootx() + (parent.winfo_width() - self.winfo_reqwidth()) // 2)
        y = max(0, parent.winfo_rooty() + (parent.winfo_height() - self.winfo_reqheight()) // 2)
        self.geometry("+%s+%s" % (x, y))
        self.grab_set()
        self.focus_set()
        self.wait_window()
        if previous_grab is not None and previous_grab.winfo_exists():
            previous_grab.grab_set()

    def _work(self):
        try:
            result = self._setup.run()
            self._prepared = result
            if self._setup.cancel_event.is_set():
                self._setup.unload(result.model)
            else:
                self._events.put(("success", result, ()))
        except SetupCancelled:
            pass
        except Exception as exc:
            self._events.put(("error", str(exc), ()))

    def _append(self, text):
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _poll(self):
        if self._closed:
            return
        while True:
            try:
                kind, value, args = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "hardware":
                self._hardware.configure(
                    text=tr("%s · %.1f GB RAM · %.1f GB free disk space")
                    % (value.cpu, value.ram_gb, value.disk_gb)
                )
            elif kind == "status":
                text = tr(value) % args if args else tr(value)
                self._status.configure(text=text)
                self._append(text)
                self._progress.stop()
                self._progress.configure(mode="indeterminate")
                self._progress.start()
            elif kind == "progress":
                total = value.get("total", 0)
                if total:
                    self._progress.stop()
                    self._progress.configure(
                        mode="determinate", maximum=total, value=value.get("completed", 0)
                    )
            elif kind == "measured":
                self._append(tr("%s: %.1f and %.1f tokens/s") % (value, *args))
            elif kind == "failed":
                self._append(tr("%s could not be prepared: %s") % (value, tr(args[0])))
            elif kind == "success":
                self._selection = value
                self._progress.stop()
                self._progress.configure(mode="determinate", maximum=100, value=100)
                self._status.configure(
                    text=tr("Ready: %s · %.1f tokens/s (slower test)")
                    % (value.model, value.tokens_per_second)
                )
                self._use.configure(state="normal")
            elif kind == "error":
                self._progress.stop()
                self._status.configure(text=tr(value))
                self._append(tr(value))
                self._cancel_button.configure(text=tr("Close"))
        self.after(100, self._poll)

    def _accept(self):
        self.result = self._selection
        self._closed = True
        self.destroy()

    def _cancel(self):
        self._closed = True
        self._setup.cancel()
        if self._prepared:
            threading.Thread(
                target=self._setup.unload, args=(self._prepared.model,), daemon=True
            ).start()
        self.destroy()
