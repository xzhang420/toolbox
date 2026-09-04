"""Small dependency-free filesystem browser for the notebook dashboard."""

from __future__ import annotations

import os
import string
from pathlib import Path


class InlinePathBrowser:
    """Browse paths on the computer running the Jupyter kernel."""

    def __init__(self, widgets):
        self.w = widgets
        self.target = None
        self.mode = "directory"
        self.extensions: tuple[str, ...] = ()
        self.on_select = None
        self.current_dir = Path.cwd()

        style = {"description_width": "initial"}
        self.title = widgets.HTML("<b>Select a path</b>")
        self.places = widgets.Dropdown(
            description="Location",
            style=style,
            layout=widgets.Layout(width="100%"),
        )
        self.location = widgets.Text(
            description="Current folder",
            style=style,
            layout=widgets.Layout(width="100%"),
        )
        self.entries = widgets.Select(
            options=(),
            rows=12,
            layout=widgets.Layout(width="100%", min_height="250px"),
        )
        self.message = widgets.HTML()
        self.up_button = self._button("Up")
        self.go_button = self._button("Go to path")
        self.open_button = self._button("Open folder")
        self.current_button = self._button("Use current folder", "primary")
        self.select_button = self._button("Use selected", "primary")
        self.cancel_button = self._button("Cancel")
        button_row = widgets.HBox(
            [
                self.up_button,
                self.go_button,
                self.open_button,
                self.current_button,
                self.select_button,
                self.cancel_button,
            ],
            layout=widgets.Layout(display="flex", flex_flow="row wrap", width="100%"),
        )
        self.widget = widgets.VBox(
            [self.title, self.places, self.location, self.entries, button_row, self.message],
            layout=widgets.Layout(
                display="none",
                width="100%",
                border="1px solid #aaa",
                padding="8px",
                margin="6px 0",
            ),
        )
        self._wire()

    def _button(self, description, button_style=""):
        width = max(100, len(description) * 8 + 28)
        return self.w.Button(
            description=description,
            button_style=button_style,
            layout=self.w.Layout(width="auto", min_width=f"{width}px", margin="2px 6px 2px 0"),
        )

    def _wire(self):
        self.up_button.on_click(self._up)
        self.go_button.on_click(self._go)
        self.open_button.on_click(self._open_selected)
        self.current_button.on_click(self._choose_current)
        self.select_button.on_click(self._choose_selected)
        self.cancel_button.on_click(lambda _: self.hide())
        self.places.observe(self._place_changed, names="value")

    @staticmethod
    def _existing_directory(value) -> Path:
        path = Path(str(value or "")).expanduser()
        if not str(value or "").strip():
            return Path.cwd().resolve()
        if path.is_file():
            return path.parent.resolve()
        candidate = path
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate.resolve() if candidate.is_dir() else Path.cwd().resolve()

    @staticmethod
    def _places(start: Path) -> tuple[tuple[str, str], ...]:
        candidates: list[tuple[str, Path]] = [
            ("Current selection", start),
            ("Home", Path.home()),
            ("Working folder", Path.cwd()),
        ]
        if os.name == "nt":
            candidates.extend(
                (f"Drive {letter}:", Path(f"{letter}:\\"))
                for letter in string.ascii_uppercase
                if Path(f"{letter}:\\").is_dir()
            )
        else:
            candidates.append(("Filesystem root", Path("/")))
            for path in (Path("/data01"), Path("/mnt"), Path("/media")):
                if path.is_dir():
                    candidates.append((str(path), path))

        options = []
        seen = set()
        for label, path in candidates:
            try:
                resolved = str(path.expanduser().resolve())
            except OSError:
                continue
            if resolved not in seen and Path(resolved).is_dir():
                seen.add(resolved)
                options.append((label, resolved))
        return tuple(options)

    def show(
        self,
        target,
        *,
        mode="directory",
        title="Select a path",
        extensions=(),
        start=None,
        on_select=None,
    ):
        if mode not in {"directory", "file", "either"}:
            raise ValueError(f"Unsupported path-browser mode: {mode}")
        self.target = target
        self.mode = mode
        self.extensions = tuple(
            extension.lower() if str(extension).startswith(".") else f".{str(extension).lower()}"
            for extension in extensions
        )
        self.on_select = on_select
        initial = start or getattr(target, "value", "") or Path.cwd()
        self.current_dir = self._existing_directory(initial)
        self.title.value = f"<b>{title}</b>"
        self.current_button.layout.display = "none" if mode == "file" else ""
        self.select_button.description = "Use selected file" if mode == "file" else "Use selected"
        self.places.options = self._places(self.current_dir)
        self.widget.layout.display = "flex"
        self._refresh(self.current_dir)
        return self.widget

    def hide(self):
        self.widget.layout.display = "none"
        self.message.value = ""

    def _refresh(self, directory):
        path = self._existing_directory(directory)
        try:
            children = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
        except (OSError, PermissionError) as exc:
            self.message.value = f"<span style='color:#c92a2a'>{type(exc).__name__}: {exc}</span>"
            return
        options = []
        for child in children:
            try:
                if child.is_dir():
                    options.append((f"[folder] {child.name}", str(child)))
                elif self.mode != "directory" and (
                    not self.extensions or child.suffix.lower() in self.extensions
                ):
                    options.append((f"[file] {child.name}", str(child)))
            except OSError:
                continue
        self.current_dir = path
        self.location.value = str(path)
        self.entries.options = tuple(options)
        self.entries.value = None
        self.message.value = f"Showing {len(options)} selectable item(s)."

    def _place_changed(self, change):
        if change.get("new") and self.widget.layout.display != "none":
            self._refresh(change["new"])

    def _up(self, _):
        self._refresh(self.current_dir.parent)

    def _go(self, _):
        raw = str(self.location.value).strip()
        path = Path(raw).expanduser()
        if path.is_dir():
            self._refresh(path)
        elif path.is_file() and self.mode in {"file", "either"}:
            self._choose(path)
        else:
            self.message.value = f"<span style='color:#c92a2a'>Path does not exist: {raw}</span>"

    def _selected(self) -> Path | None:
        return Path(self.entries.value) if self.entries.value else None

    def _open_selected(self, _):
        selected = self._selected()
        if selected is None:
            self.message.value = "Select a folder first."
        elif selected.is_dir():
            self._refresh(selected)
        elif self.mode in {"file", "either"}:
            self._choose(selected)

    def _choose_current(self, _):
        self._choose(self.current_dir)

    def _choose_selected(self, _):
        selected = self._selected()
        if selected is None:
            self.message.value = "Select an item first."
            return
        if self.mode == "directory" and not selected.is_dir():
            self.message.value = "Select a folder."
            return
        if self.mode == "file" and not selected.is_file():
            self.message.value = "Select a file."
            return
        self._choose(selected)

    def _choose(self, path: Path):
        if self.target is None:
            return
        selected = path.expanduser().resolve()
        self.target.value = str(selected)
        callback = self.on_select
        self.hide()
        if callback is not None:
            callback(selected)
