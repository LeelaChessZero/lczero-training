# Intercepting `stderr` for the App's Lifetime

This guide details how to permanently redirect `stderr` for a Textual application, capturing logs from long-running background processes like a `TrainingDaemon`.

### The Challenge

The `TrainingDaemon` runs in a background thread and logs to `stderr`. A short-lived redirection (like a `with` block) will not work. We need to redirect `stderr` for the entire duration of the app's execution.

### The Plan

1.  **Start-up Redirection:** When the app starts, permanently redirect the OS `stderr` file descriptor to a pipe.
2.  **Continuous Reading:** A background thread reads from the pipe for the app's entire lifetime.
3.  **Message Passing:** The thread posts Textual `Message`s with the captured log lines.
4.  **Shutdown Restoration:** When the app exits, restore the original `stderr`.

### Implementation Strategy

The `StderrRedirector` controller widget is still the ideal place for this logic. We will modify it to manage the redirection across its own lifecycle using `on_mount` and `on_unmount`.

1.  **Modify `StderrRedirector` for Permanent Redirection**:
    *   The `on_mount` method will set up the pipe, start the reader thread, and perform the `os.dup2` redirection.
    *   The `on_unmount` method will restore the original `stderr` file descriptor, providing clean shutdown.

    ```python
    # In StderrRedirector widget
    def on_mount(self) -> None:
        """Set up the pipe and perform the permanent redirection."""
        self._pipe_read, self._pipe_write = os.pipe()
        self._write_stream = os.fdopen(self._pipe_write, 'w')
        
        # Save original stderr and redirect
        self._original_stderr_fd = os.dup(sys.stderr.fileno())
        os.dup2(self._write_stream.fileno(), sys.stderr.fileno())

        # Start the reader thread
        self._reader_thread = threading.Thread(target=self._read_from_pipe, daemon=True)
        self._reader_thread.start()

    def on_unmount(self) -> None:
        """Close the pipe and restore original stderr on app exit."""
        self._write_stream.close() # Signals reader thread to stop
        os.dup2(self._original_stderr_fd, sys.stderr.fileno())
        os.close(self._original_stderr_fd)
    ```

2.  **Simplify `TrainingTuiApp` Integration**:
    The app's responsibility is now extremely simple: just include the `StderrRedirector` and handle its messages. The redirection is automatic.

    ```python
    class TrainingTuiApp(App):
        def compose(self) -> ComposeResult:
            yield RichLog(id="log_viewer")
            # ... other widgets ...
            yield StderrRedirector() # Automatically activates on mount

        # The message handler remains the same
        def on_stderr_redirector_line(self, message: StderrRedirector.Line):
            self.query_one("#log_viewer", RichLog).write(message.line)

        def on_ready(self) -> None:
            """
            Instantiate the daemon after the TUI is ready and
            redirection is active.
            """
            self.training_daemon = TrainingDaemon()
            self.training_daemon.start_training_loop()
    ```

### Correct Execution Flow

1.  `TrainingTuiApp` starts.
2.  `StderrRedirector` is composed and mounted. Its `on_mount` method immediately redirects `stderr`.
3.  The app's `on_ready` method fires. The `TrainingDaemon` is created.
4.  From this point on, any `stderr` output from the daemon's background thread is captured by the redirector and sent as a message to the app.
5.  The user quits the app. `StderrRedirector` is unmounted, and its `on_unmount` method cleanly restores the original `stderr`.