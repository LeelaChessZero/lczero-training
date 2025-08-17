# ABOUTME: Main TUI application class implementing the training dashboard.
# ABOUTME: Uses Textual framework to create a full-screen interface with four panes.

import subprocess
import sys
import time

import anyio
from anyio.streams.text import TextReceiveStream, TextSendStream
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Footer, Static

from ..protocol.communicator import AsyncCommunicator
from ..protocol.messages import StartTrainingPayload, TrainingStatusPayload
from .data_pipeline_pane import DataPipelinePane
from .log_pane import StreamingLogPane


class HeaderBar(Static):
    """Top header bar showing uptime and overall status."""

    def __init__(self) -> None:
        super().__init__()
        self._start_time = time.time()

    def compose(self) -> ComposeResult:
        yield Static(
            "Uptime: 00:00:00 | Status: WAITING FOR DATA", id="header-content"
        )

    def on_mount(self) -> None:
        """Start the uptime timer."""
        self.set_interval(1.0, self.update_header)

    def update_header(self) -> None:
        """Update the header with current uptime and status."""
        elapsed = int(time.time() - self._start_time)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        try:
            header_content = self.query_one("#header-content", Static)
            header_content.update(
                f"Uptime: {uptime} | Status: WAITING FOR DATA"
            )
        except NoMatches:
            pass


class TrainingStatusPane(Static):
    """Right pane showing JAX training status and metrics."""

    def compose(self) -> ComposeResult:
        yield Static(
            "JAX Training Status\n\n"
            "Training metrics will be displayed here when active:\n"
            "• Epoch Progress\n• Performance Metrics\n• Loss Values",
            classes="training-content",
        )


class TrainingTuiApp(App):
    """Main TUI application for the training dashboard.

    This creates a full-screen interface with four main panes:
    - Header bar with uptime and status
    - Data pipeline pane (main/left)
    - Training status pane (right)
    - Log pane (bottom)
    """

    CSS_PATH = "app.tcss"

    _log_stream: TextReceiveStream
    _daemon_process: anyio.abc.Process
    _communicator: AsyncCommunicator
    _config_file: str
    _data_pipeline_pane: DataPipelinePane

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, daemon_process: anyio.abc.Process):
        """Initialize the TUI app with an already-created daemon process."""
        super().__init__()
        self._daemon_process = daemon_process
        assert daemon_process.stderr is not None
        assert daemon_process.stdin is not None
        assert daemon_process.stdout is not None
        self._log_stream = TextReceiveStream(daemon_process.stderr)
        self._communicator = AsyncCommunicator(
            handler=self,
            input_stream=TextReceiveStream(daemon_process.stdout),
            output_stream=TextSendStream(daemon_process.stdin),
        )

    @classmethod
    async def create(cls, config_file: str) -> "TrainingTuiApp":
        """Create a new TrainingTuiApp with async subprocess creation."""
        daemon_process = await anyio.open_process(
            [sys.executable, "-m", "lczero_training.daemon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        app = cls(daemon_process)
        app._config_file = config_file
        return app

    def compose(self) -> ComposeResult:
        """Compose the main UI layout."""
        yield HeaderBar()

        self._data_pipeline_pane = DataPipelinePane()
        self._data_pipeline_pane.border_title = "Training data pipeline"
        yield self._data_pipeline_pane
        training_status_pane = TrainingStatusPane()
        training_status_pane.border_title = "Training Status"
        yield training_status_pane
        yield StreamingLogPane(stream=self._log_stream)

        yield Footer()

    def on_mount(self) -> None:
        """Start the communicator when the app mounts."""
        self.run_worker(self._communicator.run(), exclusive=True)
        self.run_worker(self._send_start_training(), exclusive=False)

    async def _send_start_training(self) -> None:
        """Send StartTrainingPayload with the config file."""
        payload = StartTrainingPayload(config_filepath=self._config_file)
        await self._communicator.send(payload)

    def action_quit(self) -> None:  # type: ignore
        """Handle quit action."""
        self._daemon_process.terminate()
        self.exit()

    async def on_training_status(self, payload: TrainingStatusPayload) -> None:
        """Handle training status updates."""
        self._data_pipeline_pane.update_metrics(
            payload.dataloader_1_second, payload.dataloader_total
        )
