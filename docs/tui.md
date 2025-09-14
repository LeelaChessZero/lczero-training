# UI Design

The application will present a single-screen dashboard with a classic blue background, organized into several key, always-visible panes.

## 1. Overall Layout

The screen is divided into four main sections:

1. **Header Bar (Top):** A slim, single-line bar at the very top.
2. **Data Pipeline Pane (Top-Left/Main):** The largest and most detailed pane.
3. **JAX Training Status Pane (Right):** A pane dedicated to live training metrics.
4. **Log Pane (Bottom):** A pane for displaying raw log output.

## 2. Header Bar

This bar provides high-level, global status at a glance.

- **Uptime:** Total wall-clock time since the script was launched.
- **Overall Stage:** The current high-level state of the application. Will display one of: `WAITING FOR DATA`, `TRAINING`, `EXPORTING`, or `ERROR`.

### 3. Data Pipeline Pane

This pane visualizes the flow of data through the C++ pipeline. It will use a responsive flow layout, where widgets are arranged side-by-side and wrap to the next line if space is insufficient.

- **Pipeline Stages (Blocks):** Each stage of the C++ pipeline is a distinct widget.
  - **Load Meter:** Most stages will show the number of active worker threads vs. allocated threads (e.g., `Load: 3.4/4`).
  - **Specific Stats:** More complex stages will display additional, specific information.
    - `FilePathProvider`: Stage (e.g., `Initial Scan`, `Watching`), Total files found.
    - `ShufflingChunkPool`: A **full-width widget**. Will display:
      - Initial fill progress as a progress bar.
      - Current pool size and target size.
      - A count of "buffer exhausted" events.
      - Distribution of chunks in the buffer among different training epochs.
    - `ChunkValidator`: Number and percentage of invalid chunks found, broken down by reason.

- **Queues (Connectors):** Between each stage widget, a queue widget will show the state of the message queue connecting them.
  - **Title:** Name of the queue.
  - **Fullness Text:** e.g., `850/1000`.
  - **Fullness Bar:** A color-coded pseudographic bar (`██████░░░░`) to
    visualize fullness. The color indicates the health of the queue (e.g., green
    for consumer-bound, red for producer-bound).
  - **Throughput:** The current rate of items passing through, e.g.,
    `12.3k items/s`.

- **Train/Validation/Test Splitter:**
  - The pipeline view will show a "Stream Splitter" stage.
  - Hotkeys (e.g., F1, F2, F3) will allow the user to instantly switch the view
  - After this stage, the UI will display the pipeline stats for **one** stream
    at a time (defaulting to 'Training'). to show the stats for the 'Training',
    'Validation', or 'Test' streams.

### 4. Training schedule/pipeline pane

The area below the Data Pipeline Pane is split horizontally into two
sections: Training Schedule (left) and JAX Training Status (right).

The Training Schedule pane shows:

- **Combined uptime and stage line**: "Uptime: 2d 14:30:45   Stage: TRAINING"
  - Uptime includes days when >24 hours (format: "2d 14:30:45")
  - Stage shows current training state (WAITING_FOR_DATA, TRAINING, EXPORTING, ERROR)
- **Completed epochs**: Simple counter of epochs completed since daemon start
- **New chunks progress bar**: Shows chunks collected since training start vs. target
  - Indeterminate state when target is unknown (0)
- **Training time progress bar**: Current training time vs. previous training duration
  - Indeterminate state when no previous duration exists
- **Cycle time progress bar**: Current cycle time vs. previous cycle duration
  - Indeterminate state when no previous duration exists

**Implementation details:**

- **Header bar**: Completely empty (no content)
- **Data structure**: `TrainingScheduleData` dataclass with all timing fields:
  - `current_stage: TrainingStage` (enum)
  - `completed_epochs_since_start: int`
  - `new_chunks_since_training_start: int`
  - `chunks_to_wait: int`
  - `total_uptime_seconds: float`
  - `current_training_time_seconds: float`
  - `previous_training_time_seconds: float`
  - `current_cycle_time_seconds: float`
  - `previous_cycle_time_seconds: float`
- **Timing computation**: All timing values computed in daemon, not TUI
- **Progress bars**: Show indeterminate state when maximum values ≤ 0
- **Layout**: Compact single-line widgets with no extra padding/margins
- **Files**:
  - `training_widgets.py`: Widget implementations
  - `pipeline.py`: Training state tracking and data collection
  - `daemon.py`: Metrics collection and transmission
  - `messages.py`: Protocol definitions with enum serialization support

### 5. JAX Training Status Pane

This pane is dedicated to the live status of an active JAX training run. It
remains blank or shows summary info when the system is not actively training.

- **Epoch Progress:** A progress bar showing completion of the current epoch
  (`Step 12345 / 50000`).
- **Performance Metrics:**
  - Steps per second: `345.6 steps/s`.
  - Estimated Time Remaining (ETR) for the current run.
  - Total wall time spent on the current training run.
- **Loss Values (Numerical Only):**
  - A prominent display of the **Total Loss**.
  - A compact 2-column grid displaying the individual values for the **7 Head
    Losses**.

### 6. Log Pane

A pane across the bottom of the screen.

- **Content:** A direct feed of all output sent to `stderr` from any part of the
  application (Python or C++).
- **Functionality:** The pane will be scrollable and will hold a fixed number of
  lines (e.g., 1000) to prevent unbounded memory usage, discarding the oldest
  lines as new ones arrive.

