"""Small reusable Tk widgets used by the desktop setup form."""

import tkinter as tk
from tkinter import ttk


class ScrollablePanel(ttk.Frame):
    def __init__(self, parent, background, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, background=background, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.content = ttk.Frame(self.canvas, style="Panel.TFrame", padding=22)
        self._window = self.canvas.create_window(0, 0, window=self.content, anchor="nw")
        self.canvas.bind("<Configure>", self._resize)
        self.content.bind("<Configure>", self._update_region)

    def _resize(self, event):
        self.canvas.itemconfigure(self._window, width=max(1, event.width))

    def _update_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def bind_navigation(self):
        def bind(widget):
            # Leave editable/choice controls' own wheel behavior intact.
            if widget.winfo_class() not in {"TCombobox", "TEntry", "Text"}:
                widget.bind("<MouseWheel>", self._wheel, add="+")
            widget.bind("<FocusIn>", lambda event: self.ensure_visible(event.widget), add="+")
            for child in widget.winfo_children():
                bind(child)
        self.canvas.bind("<MouseWheel>", self._wheel)
        bind(self.content)

    def _wheel(self, event):
        if self.content.winfo_height() > self.canvas.winfo_height() and event.delta:
            steps = max(1, abs(event.delta) // 120)
            self.canvas.yview_scroll(-steps if event.delta > 0 else steps, "units")
        return "break"

    def ensure_visible(self, widget):
        """Keep keyboard-focused fields inside the visible canvas viewport."""
        height = self.content.winfo_height()
        viewport = self.canvas.winfo_height()
        if height <= viewport:
            return
        top = widget.winfo_rooty() - self.content.winfo_rooty()
        bottom = top + widget.winfo_height()
        current = self.canvas.canvasy(0)
        if top < current:
            self.canvas.yview_moveto(max(0, top - 8) / height)
        elif bottom > current + viewport:
            self.canvas.yview_moveto(max(0, bottom - viewport + 8) / height)
