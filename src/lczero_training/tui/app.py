# ABOUTME: Main TUI application class implementing the training dashboard.
# ABOUTME: Uses Textual framework to create a full-screen interface with four panes.

import argparse
import signal
import subprocess
import sys
from typing import Iterable, Optional

import anyio
from anyio.streams.text import TextReceiveStream, TextSendStream
from textual.app import App, ComposeResult, SystemCommand
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Static

from ..daemon.protocol.communicator import AsyncCommunicator
from ..daemon.protocol.messages import (
    StartTrainingImmediatelyPayload,
    StartTrainingPayload,
    TrainingStatusPayload,
)
from .data_pipeline_pane import DataPipelinePane
from .log_pane import StreamingLogPane
from .training_widgets import TrainingScheduleWidget


class HeaderBar(Static):
    """Empty header bar."""

    def compose(self) -> ComposeResult:
        return
        yield  # unreachable, but makes the function a generator


class JAXTrainingPane(Static):
    """Right pane showing JAX training status and metrics."""

    def compose(self) -> ComposeResult:
        yield Static(
            "JAX Training Status\n\n"
            "Live training metrics will be displayed here when active:\n"
            "• Epoch Progress\n• Performance Metrics\n• Loss Values",
            classes="jax-training-content",
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

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """Adds all required command-line arguments to the given parser."""
        parser.add_argument(
            "--config",
            required=True,
            help="Path to the training configuration file",
        )
        parser.add_argument(
            "--logfile",
            help="Path to the log file for saving TUI output",
        )

    _log_stream: TextReceiveStream
    _daemon_process: anyio.abc.Process
    _communicator: AsyncCommunicator
    _config_file: str
    _logfile: Optional[str]
    _data_pipeline_pane: DataPipelinePane
    _training_schedule_widget: TrainingScheduleWidget

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, args: Optional[argparse.Namespace] = None) -> None:
        """
        Initializes the app.
        If 'args' is provided, it's used directly.
        If 'args' is None, fallback to parsing sys.argv.
        """
        super().__init__()

        if args is None:
            # Fallback for when run by "textual run"
            parser = argparse.ArgumentParser()
            TrainingTuiApp.add_arguments(parser)
            args, _ = parser.parse_known_args()

        # Consume configuration from the args object
        self._config_file: str = args.config
        self._logfile: Optional[str] = args.logfile

    async def on_load(self) -> None:
        """Start the daemon process and communicator when the app loads."""
        # Create the daemon process
        self._daemon_process = await anyio.open_process(
            [sys.executable, "-m", "lczero_training.daemon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        assert self._daemon_process.stderr is not None
        assert self._daemon_process.stdin is not None
        assert self._daemon_process.stdout is not None

        # Set up streams and communicator
        self._log_stream = TextReceiveStream(self._daemon_process.stderr)
        self._communicator = AsyncCommunicator(
            handler=self,
            input_stream=TextReceiveStream(self._daemon_process.stdout),
            output_stream=TextSendStream(self._daemon_process.stdin),
        )

    def compose(self) -> ComposeResult:
        """Compose the main UI layout."""
        yield HeaderBar()

        self._data_pipeline_pane = DataPipelinePane()
        self._data_pipeline_pane.border_title = "Training data pipeline"
        yield self._data_pipeline_pane

        # Horizontal split below the data pipeline pane
        with Horizontal(id="training-status-container"):
            self._training_schedule_widget = TrainingScheduleWidget()
            self._training_schedule_widget.border_title = "Training Schedule"
            yield self._training_schedule_widget

            jax_training_pane = JAXTrainingPane()
            jax_training_pane.border_title = "JAX Training Status"
            yield jax_training_pane

        yield StreamingLogPane(
            stream=self._log_stream, logfile_path=self._logfile
        )
        yield Footer()

    def on_mount(self) -> None:
        """Start the communicator when the app mounts."""
        self.run_worker(self._communicator.run(), exclusive=True)
        self.run_worker(self._send_start_training(), exclusive=False)

    async def _send_start_training(self) -> None:
        """Send StartTrainingPayload with the config file."""
        payload = StartTrainingPayload(config_filepath=self._config_file)
        await self._communicator.send(payload)

    async def _command_start_training_immediately(self) -> None:
        """Trigger immediate training without waiting for additional chunks."""

        payload = StartTrainingImmediatelyPayload()
        await self._communicator.send(payload)
        self.notify("Requested immediate training start.")

    async def action_quit(self) -> None:  # type: ignore
        """Handle quit action."""
        self._daemon_process.send_signal(signal.SIGINT)
        with anyio.move_on_after(20) as scope:
            await self._daemon_process.wait()
        if scope.cancelled_caught:
            self._daemon_process.terminate()
        self.exit()

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        """Add application-specific commands to the command palette."""

        yield from super().get_system_commands(screen)
        yield SystemCommand(
            "Start training immediately",
            "Begin the next training cycle without waiting for chunks.",
            self._command_start_training_immediately,
        )

    async def on_training_status(self, payload: TrainingStatusPayload) -> None:
        """Handle training status updates."""
        self._data_pipeline_pane.update_metrics(
            payload.dataloader_1_second, payload.dataloader_total
        )

        # Update training schedule widget
        self._training_schedule_widget.update_training_schedule(
            payload.training_schedule
        )
