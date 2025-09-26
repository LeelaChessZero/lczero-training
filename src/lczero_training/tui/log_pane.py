import datetime
from pathlib import Path
from typing import Any, Optional, TextIO

from anyio.streams.text import TextReceiveStream
from textual.widgets import RichLog


class StreamingLogPane(RichLog):
    """Log pane that streams output from an async text stream."""

    def __init__(
        self,
        stream: TextReceiveStream,
        logfile_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            highlight=True, markup=True, max_lines=1000, wrap=True, **kwargs
        )
        self._stream = stream
        self._logfile_path = logfile_path
        self._logfile: Optional[Path] = None
        self._logfile_handle: Optional[TextIO] = None
        if logfile_path:
            self._logfile = Path(logfile_path)

    def on_mount(self) -> None:
        """Start the async reading task when the widget is mounted."""
        if self._logfile:
            self._write_banner()
        self.run_worker(self._read_stream())

    def _write_banner(self) -> None:
        """Write a session banner to the logfile."""
        if not self._logfile:
            return

        self._logfile.parent.mkdir(parents=True, exist_ok=True)
        self._logfile_handle = self._logfile.open("a", encoding="utf-8")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        banner = (
            f"\n{'=' * 80}\n"
            f"LCZero Training TUI Session Started: {timestamp}\n"
            f"{'=' * 80}\n"
        )
        self._logfile_handle.write(banner)
        self._logfile_handle.flush()

    def _write_to_file(self, line: str) -> None:
        """Write a line to the logfile."""
        if not self._logfile_handle:
            return

        self._logfile_handle.write(f"{line}\n")
        self._logfile_handle.flush()

    async def _read_stream(self) -> None:
        """Async function that reads lines from the text stream."""
        try:
            async for line in self._stream:
                line = line.strip()
                if line:
                    self.write(line)
                    self._write_to_file(line)
        except Exception:
            pass
