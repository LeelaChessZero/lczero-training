# ABOUTME: CLI entry point for running TrainingDaemon as subprocess.
# ABOUTME: Creates and starts daemon instance for IPC communication with parent TUI.

from .daemon import TrainingDaemon


def main():
    daemon = TrainingDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
