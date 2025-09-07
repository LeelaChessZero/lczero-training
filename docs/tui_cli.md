
### **Specification: Self-Configuring Textual App Subcommand**

**Goal:** Integrate a `Textual` application as a subcommand within a larger `argparse`-based CLI. The solution must be elegant, robust, and adhere to the DRY (Don't Repeat Yourself) principle.

**Core Principle:** The `Textual` App class will be the **single source of truth** for its own command-line arguments. It achieves this by defining its arguments in a `staticmethod` and consuming a parsed `argparse.Namespace` object in its constructor. This allows it to be configured directly by our CLI (via dependency injection) or to configure itself when run by an external tool like `textual run` (via a fallback mechanism).

---

#### 1. File Structure

The relevant files for this task are organized as follows:

```
my_app/
└── tui/
    ├── __init__.py      # Empty
    ├── app.py           # Contains the TrainingTuiApp class
    └── __main__.py      # The subcommand dispatcher
```

---

#### 2. Implementation Details

##### **File A: The App Class (`tui/app.py`)**

**Role:** This file contains the `TrainingTuiApp` class, which is the "source of truth." It defines its CLI argument needs and knows how to initialize itself from them.

**Contract:**
1.  A `staticmethod` `add_arguments(parser)` that populates a given `ArgumentParser`.
2.  An `__init__(self, args=None)` that accepts an optional `argparse.Namespace` object.
3.  If `args` is `None`, it must parse the arguments itself using `parse_known_args()`.

**Code:**
```python
# In my_app/tui/app.py
import argparse
from typing import Optional
import textual.app

class TrainingTuiApp(textual.app.App):
    """A self-configuring Textual app."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """Adds all required command-line arguments to the given parser."""
        parser.add_argument(
            "--config",
            required=True,
            help="Path to the training configuration file",
        )
        # Future arguments for this app go here.

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

    def on_mount(self) -> None:
        self.log(f"App mounted. Config: {self._config_file}")
```

---

##### **File B: The Subcommand Dispatcher (`tui/__main__.py`)**

**Role:** This file acts as a "pure conduit." It connects the main CLI to the `TrainingTuiApp` without knowing the details of its arguments.

**Contract:**
1.  `configure_parser(parser)` must delegate argument definition by calling `TrainingTuiApp.add_arguments()`.
2.  `run(args)` must instantiate `TrainingTuiApp`, passing the entire `args` object to its constructor.

**Code:**
```python
# In my_app/tui/__main__.py
import argparse
from .app import TrainingTuiApp

def configure_parser(parser: argparse.ArgumentParser):
    """Delegates argument configuration to the app class."""
    TrainingTuiApp.add_arguments(parser)
    parser.set_defaults(func=run)

def run(args: argparse.Namespace):
    """Instantiates and runs the app, injecting the parsed args."""
    app = TrainingTuiApp(args=args)
    app.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone TUI runner")
    configure_parser(parser)
    args = parser.parse_args()
    args.func(args)
```

---

#### 3. How It Works: Two Execution Paths

1.  **Via Your Main CLI** (`python -m my_app tui --config ...`):
    *   The root `argparse` calls `tui.__main__.configure_parser`.
    *   This calls `TrainingTuiApp.add_arguments`, adding `--config` to the parser.
    *   After parsing, the complete `args` object is passed to `tui.__main__.run`.
    *   `run` instantiates `TrainingTuiApp(args=args)`, injecting the configuration. The app's `__init__` sees `args` is not `None` and uses it directly.

2.  **Via Textual's Runner** (`textual run --dev my_app.tui.app:TrainingTuiApp --config ...`):
    *   `textual` instantiates the app with no parameters: `TrainingTuiApp()`.
    *   The app's `__init__` sees that `args` *is* `None`, triggering the fallback.
    *   It creates a temporary parser, calls its own `add_arguments`, and uses `parse_known_args()` to safely find `--config` while ignoring `textual`'s `--dev` flag.

---

#### 4. Verification Steps

To confirm the implementation is correct, run the following commands:

1.  **Test with the main CLI:**
    ```bash
    python -m my_app tui --config /path/to/your/config.file
    ```
    *   **Expected:** The Textual TUI should launch successfully and log the correct config file path.

2.  **Test with the Textual runner:**
    ```bash
    textual run --dev my_app.tui.app:TrainingTuiApp --config /path/to/your/config.file
    ```
    *   **Expected:** The Textual TUI should launch successfully in development mode and log the correct config file path.