import threading
from typing import IO, Optional

from textual.widgets import RichLog


class StreamingLogPane(RichLog):
    """Log pane that streams output from a file-like object in a background thread."""

    def __init__(self, stream: Optional[IO[str]] = None, **kwargs) -> None:
        """Initialize the streaming log pane.

        Args:
            stream: File-like object to read from (e.g., subprocess stderr).
            **kwargs: Additional arguments passed to RichLog.
        """
        super().__init__(highlight=True, markup=True, max_lines=1000, **kwargs)
        self._stream = stream
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def on_mount(self) -> None:
        """Start the background thread when the widget is mounted."""
        if self._stream is not None:
            self._stop_event.clear()
            self._reader_thread = threading.Thread(
                target=self._read_stream, daemon=True
            )
            self._reader_thread.start()

    def on_unmount(self) -> None:
        """Stop the background thread when the widget is unmounted."""
        if self._reader_thread is not None:
            self._stop_event.set()
            self._reader_thread.join(timeout=1.0)

    def _read_stream(self) -> None:
        """Background thread function that reads from the stream."""
        if self._stream is None:
            return

        try:
            while not self._stop_event.is_set():
                line = self._stream.readline()
                if not line:
                    break

                line = line.rstrip("\n\r")
                if line:
                    self.app.call_from_thread(self.write, line)

        except Exception:
            pass
