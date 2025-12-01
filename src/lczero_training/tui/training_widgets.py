from textual.app import ComposeResult
from textual.widgets import ProgressBar, Static

from ..daemon.protocol.messages import TrainingScheduleData


class TimeProgressWidget(Static):
    """A widget to display a label, progress bar, and time ratio."""

    def __init__(self, label: str, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._label = label

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="time-label")
        yield ProgressBar(show_eta=False, classes="time-progress-bar")
        yield Static("", classes="time-ratio")

    def update_progress(
        self,
        current: float,
        total: float,
        current_formatted: str | None = None,
        total_formatted: str | None = None,
    ) -> None:
        """Update the progress bar and ratio text."""
        progress_bar = self.query_one(ProgressBar)

        progress_bar.total = total or None
        progress_bar.progress = current

        current_str = (
            current_formatted
            if current_formatted is not None
            else str(int(current))
        )
        total_str = (
            total_formatted if total_formatted is not None else str(int(total))
        )

        self.query_one(".time-ratio", Static).update(
            f"{current_str}/{total_str}"
        )


def format_time_duration(seconds: float) -> str:
    """Format time duration in seconds to human readable format with days support."""
    if seconds <= 0:
        return "--"

    total_seconds = int(seconds)
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class TrainingScheduleWidget(Static):
    """A widget to display training schedule information."""

    def compose(self) -> ComposeResult:
        yield Static("Uptime: --   Stage: --", id="uptime-stage-display")
        yield Static("Completed epochs: 0", id="epochs-display")
        yield TimeProgressWidget("New Chunks:", id="chunks-progress")
        yield TimeProgressWidget("Training time", id="training-time-progress")
        yield TimeProgressWidget("Cycle time", id="cycle-time-progress")

    def update_training_schedule(
        self, data: TrainingScheduleData | None
    ) -> None:
        """Update the widget with new training schedule data."""
        if not data:
            return

        uptime_str = format_time_duration(data.total_uptime_seconds)
        self.query_one("#uptime-stage-display", Static).update(
            f"Uptime: {uptime_str}   Stage: {data.current_stage.value}"
        )

        self.query_one("#epochs-display", Static).update(
            f"Completed epochs: {data.completed_epochs_since_start}"
        )

        self.query_one("#chunks-progress", TimeProgressWidget).update_progress(
            current=data.new_chunks_since_training_start,
            total=data.chunks_to_wait,
        )

        self.query_one(
            "#training-time-progress", TimeProgressWidget
        ).update_progress(
            current=data.current_training_time_seconds,
            total=data.previous_training_time_seconds,
            current_formatted=format_time_duration(
                data.current_training_time_seconds
            ),
            total_formatted=format_time_duration(
                data.previous_training_time_seconds
            ),
        )

        self.query_one(
            "#cycle-time-progress", TimeProgressWidget
        ).update_progress(
            current=data.current_cycle_time_seconds,
            total=data.previous_cycle_time_seconds,
            current_formatted=format_time_duration(
                data.current_cycle_time_seconds
            ),
            total_formatted=format_time_duration(
                data.previous_cycle_time_seconds
            ),
        )
