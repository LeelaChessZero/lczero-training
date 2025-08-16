from anyio.streams.text import TextReceiveStream
from textual.widgets import RichLog


class StreamingLogPane(RichLog):
    """Log pane that streams output from an async text stream."""

    def __init__(self, stream: TextReceiveStream, **kwargs) -> None:
        super().__init__(highlight=True, markup=True, max_lines=1000, **kwargs)
        self._stream = stream

    def on_mount(self) -> None:
        """Start the async reading task when the widget is mounted."""
        self.run_worker(self._read_stream())

    async def _read_stream(self) -> None:
        """Async function that reads lines from the text stream."""
        try:
            async for line in self._stream:
                line = line.strip()
                if line:
                    self.write(line)
        except Exception:
            pass
