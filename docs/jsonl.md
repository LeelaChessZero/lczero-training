# Bi-directional JSONL IPC Protocol

## **1. Overview**

The goal is to create a simple, elegant, and robust system for two Python processes (a parent and a child) to communicate. Communication will occur over the child process's `stdin` and `stdout` streams using the JSON Lines (`.jsonl`) format.

The system will facilitate bi-directional "notifications" rather than a request/response RPC model. Each notification consists of a string identifier (`type`) and a structured `payload` (a Python dataclass). The implementation should be idiomatic, concise, and avoid over-engineering.

## **2. Core Concepts**

*   **Notification:** A single, one-way message sent from one process to another. It does not expect a direct response.
*   **Event Type:** A unique string identifier for a kind of notification (e.g., `"training_started"`).
*   **Payload:** A Python `dataclass` containing the structured data for a specific event type. Each event type has its own payload dataclass.
*   **Handler:** A method within a user-defined class that is automatically called when a specific notification is received.

## **3. On-the-Wire Protocol**

All communication between processes must be in the JSON Lines format. Each line sent to a stream must be a complete, self-contained JSON object terminated by a newline character (`\n`).

Each JSON object **must** have the following structure:

```json
{"type": "event_type_string", "payload": {...}}
```

*   `type`: A string literal that identifies the event.
*   `payload`: A JSON object containing the data for the event. The structure of this object corresponds to the fields of the associated payload dataclass.

**Example:**
```json
{"type": "epoch_complete", "payload": {"epoch": 5, "validation_loss": 0.123}}
```

## **4. Python Implementation Architecture**

The implementation will be organized into a `protocol` package containing three key modules.

## **4.1. Proposed File Structure**

```
my_project/
├── parent_process.py        # Main application script (e.g., UI)
├── child_process.py         # Subprocess script (e.g., NN trainer)
└── protocol/
    ├── __init__.py          # Can be empty
    ├── registry.py          # Defines the registration system
    ├── messages.py          # Defines all payload dataclasses
    └── communicator.py      # Defines the core Communicator class
```

## **4.2. Event & Payload Definition (`registry.py` and `messages.py`)**

Payloads are defined as standard Python `dataclasses`. A decorator-based registration system will be used to link an `event_type` string to its corresponding payload class.

**`protocol/registry.py`:**
This module will manage the mapping between event type strings and payload classes. It should be implemented once and not require further modification. It must maintain both a forward (`string -> class`) and a reverse (`class -> string`) mapping.

```python
# protocol/registry.py
import inspect

# These maps will be populated by the @register decorator
TYPE_TO_CLASS_MAP = {}
CLASS_TO_TYPE_MAP = {}

def register(event_type: str):
    """A decorator to register a payload dataclass with its event type string."""
    def decorator(cls):
        if not inspect.isclass(cls):
             raise TypeError("The @register decorator can only be used on classes.")
        
        if event_type in TYPE_TO_CLASS_MAP:
            raise ValueError(f"Event type '{event_type}' is already registered.")
        
        if cls in CLASS_TO_TYPE_MAP:
            raise ValueError(f"Class '{cls.__name__}' is already registered.")

        TYPE_TO_CLASS_MAP[event_type] = cls
        CLASS_TO_TYPE_MAP[cls] = event_type
        return cls
    return decorator
```

**`protocol/messages.py`:**
This module is where the developer defines all payloads used in the application.

```python
# protocol/messages.py
from dataclasses import dataclass
from .registry import register

# --- Notifications from Trainer (Child) to UI (Parent) ---

@register("training_started")
@dataclass
class TrainingStartedPayload:
    total_epochs: int
    batch_size: int

@register("epoch_complete")
@dataclass
class EpochCompletePayload:
    epoch: int
    validation_loss: float

# --- Notifications from UI (Parent) to Trainer (Child) ---

@register("pause_training")
@dataclass
class PauseTrainingPayload:
    pass # No data needed for this event

@register("set_learning_rate")
@dataclass
class SetLearningRatePayload:
    new_lr: float
```

## **4.3. Event Handling**

A user-defined handler class will contain the logic to be executed upon receiving a notification. The `Communicator` will dispatch events to methods on this class based on a naming convention.

*   **Convention:** For an event with type `"some_event"`, the `Communicator` will look for a handler method named `on_some_event`.
*   **Method Signature:** The handler method must accept a single argument: the instantiated payload dataclass object.

**Example Handler:**

```python
# In parent_process.py or child_process.py
from protocol import messages

class TrainerEventHandler:
    def on_pause_training(self, payload: messages.PauseTrainingPayload):
        print("Received request to pause training.")
        # ... logic to set a pause flag ...

    def on_set_learning_rate(self, payload: messages.SetLearningRatePayload):
        print(f"Setting learning rate to: {payload.new_lr}")
        # ... logic to update the optimizer ...
```

## **4.4. The `Communicator` Class (`communicator.py`)**

This is the central component that manages communication.

```python
# protocol/communicator.py
import sys
import json
from .registry import TYPE_TO_CLASS_MAP, CLASS_TO_TYPE_MAP

class Communicator:
    def __init__(self, handler, input_stream=sys.stdin, output_stream=sys.stdout):
        """
        Initializes the Communicator.
        
        Args:
            handler: An object with `on_<event_type>` methods.
            input_stream: A file-like object to read incoming messages from.
            output_stream: A file-like object to write outgoing messages to.
        """
        self.handler = handler
        self.input = input_stream
        self.output = output_stream

    def send(self, payload_instance):
        """
        Serializes and sends a payload object as a notification.
        The event type is automatically looked up from the registry.
        """
        payload_cls = type(payload_instance)
        event_type = CLASS_TO_TYPE_MAP.get(payload_cls)
        
        if event_type is None:
            raise TypeError(f"Object of type {payload_cls.__name__} is not a registered payload.")
            
        # Convert dataclass to dict, then wrap in the protocol structure
        # Note: requires Python 3.7+ for json.dumps on dataclasses
        from dataclasses import asdict
        payload_dict = asdict(payload_instance)
        
        message = {"type": event_type, "payload": payload_dict}
        
        json.dump(message, self.output)
        self.output.write('\n')
        self.output.flush()

    def run(self):
        """
        Starts the blocking listener loop.
        Reads from the input stream line-by-line, deserializes notifications,
        and dispatches them to the appropriate handler method.
        
        This method blocks until the input stream is closed.
        """
        for line in self.input:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            event_type = data['type']
            payload_dict = data['payload']
            
            payload_cls = TYPE_TO_CLASS_MAP[event_type]
            payload_instance = payload_cls(**payload_dict)
            
            handler_method_name = f"on_{event_type}"
            handler_method = getattr(self.handler, handler_method_name)
            
            handler_method(payload_instance)
```

## **5. Error Handling Strategy**

The system will adopt a **fail-fast** strategy. Simplicity and immediate feedback on errors are prioritized over graceful recovery.

If the `Communicator.run()` method encounters any of the following on its input stream, it should **let the exception propagate and crash the process**:

1.  A line that is not valid JSON (`json.JSONDecodeError`).
2.  A valid JSON object that is missing the required `"type"` key (`KeyError`).
3.  An `event_type` string that has not been registered (`KeyError` on `TYPE_TO_CLASS_MAP` lookup).
4.  A payload that does not match the fields of its registered dataclass (`TypeError` on `**payload_dict`).

This behavior is the default when the `try...except` blocks are omitted from the `run` loop, which is the intended implementation.

## **6. Example Usage**

The application developer is responsible for launching the subprocess, wiring up the streams, and deciding on the threading model.

**`child_process.py` (The NN Trainer):**
```python
# Simplified example
import time
from protocol.communicator import Communicator
from protocol import messages

class TrainerEventHandler:
    def on_pause_training(self, payload: messages.PauseTrainingPayload):
        print("CHILD: Pausing training...", flush=True)

class Trainer:
    def train(self):
        comm = Communicator(TrainerEventHandler())
        # The main thread of the child process will be dedicated to running
        # the training loop. We run the communicator in a background thread.
        # Alternatively, the training loop could be in a thread, and comm.run()
        # could block the main thread. User choice.
        
        # This example assumes the child's primary job is training.
        # It sends notifications but only listens for them if run in a thread.
        # For this example, we'll just send.
        
        comm.send(messages.TrainingStartedPayload(total_epochs=10, batch_size=64))
        for i in range(10):
            print(f"CHILD: Training epoch {i+1}...", flush=True)
            time.sleep(1)
            comm.send(messages.EpochCompletePayload(epoch=i+1, validation_loss=1.0/(i+1)))

if __name__ == "__main__":
    Trainer().train()
```

**`parent_process.py` (The UI):**
```python
# Simplified example
import sys
import subprocess
import threading
from protocol.communicator import Communicator
from protocol import messages

class UiEventHandler:
    def on_training_started(self, payload: messages.TrainingStartedPayload):
        print(f"PARENT: Training started! Total epochs: {payload.total_epochs}")

    def on_epoch_complete(self, payload: messages.EpochCompletePayload):
        print(f"PARENT: Epoch {payload.epoch} done. Loss: {payload.validation_loss:.3f}")

# Function to read and log stderr from the subprocess
def log_stderr(pipe):
    for line in iter(pipe.readline, ''):
        print(f"[SUBPROCESS STDERR] {line.strip()}", file=sys.stderr)
    pipe.close()

if __name__ == "__main__":
    # Launch the child process
    process = subprocess.Popen(
        [sys.executable, 'child_process.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, # Use text mode for automatic encoding/decoding
        bufsize=1  # Line-buffered
    )

    # Start a thread to monitor the subprocess's stderr
    stderr_thread = threading.Thread(target=log_stderr, args=(process.stderr,), daemon=True)
    stderr_thread.start()

    handler = UiEventHandler()
    comm = Communicator(handler, input_stream=process.stdout, output_stream=process.stdin)

    # In a real UI app (like Textual), you would run comm.run() in a worker thread.
    # For this simple script, we'll run it in a thread and send a message from the main thread.
    comm_thread = threading.Thread(target=comm.run, daemon=True)
    comm_thread.start()
    
    print("PARENT: Main thread is free. Waiting a bit before sending a command...")
    time.sleep(2.5) # Wait for a couple of epochs
    
    print("PARENT: Sending pause command.")
    comm.send(messages.PauseTrainingPayload())
    
    # Wait for the process to finish
    process.wait()
    comm_thread.join(timeout=1)
```

---

## **Updated Specification Section: Nested Payloads & Deserialization**

This section supersedes parts of the original `communicator.py` and `messages.py` definitions to add support for nested dataclasses.

## **1. Example of a Nested Payload (`messages.py`)**

The `messages.py` file can now define and use nested structures.

```python
# protocol/messages.py (Updated Example)
from dataclasses import dataclass
from typing import List, Dict
from .registry import register

# A nested dataclass that does not need to be registered itself
@dataclass
class ModelConfig:
    name: str
    params: Dict[str, any]

# --- Notifications from Trainer (Child) to UI (Parent) ---

@register("training_started")
@dataclass
class TrainingStartedPayload:
    total_epochs: int
    config: ModelConfig  # <-- Nested dataclass field
    sample_data_ids: List[int] # <-- List field
```

## **2. The `Communicator` Class (`communicator.py`) - Revised**

To handle the instantiation of these nested structures, we will add a private helper function to `communicator.py`. This function will be responsible for the recursive deserialization. The public API of the `Communicator` class remains unchanged.

```python
# protocol/communicator.py (Revised)
import sys
import json
import inspect
from dataclasses import dataclass, is_dataclass, asdict
from typing import get_origin, get_args

from .registry import TYPE_TO_CLASS_MAP, CLASS_TO_TYPE_MAP

def _from_dict(cls, data):
    """
    Recursively constructs a dataclass instance from a dictionary.
    Handles nested dataclasses and lists of dataclasses.
    """
    if not is_dataclass(cls):
        return data

    constructor_args = {}
    for field in cls.__dataclass_fields__.values():
        field_value = data.get(field.name)
        if field_value is None:
            continue
            
        # Handle lists of dataclasses
        origin_type = get_origin(field.type)
        if origin_type is list or origin_type is list:
            list_item_type = get_args(field.type)[0]
            if is_dataclass(list_item_type):
                constructor_args[field.name] = [_from_dict(list_item_type, item) for item in field_value]
            else:
                constructor_args[field.name] = field_value
        # Handle nested dataclasses
        elif is_dataclass(field.type):
            constructor_args[field.name] = _from_dict(field.type, field_value)
        # Handle primitives, dicts, etc.
        else:
            constructor_args[field.name] = field_value
            
    return cls(**constructor_args)


class Communicator:
    def __init__(self, handler, input_stream=sys.stdin, output_stream=sys.stdout):
        self.handler = handler
        self.input = input_stream
        self.output = output_stream

    def send(self, payload_instance):
        """
        Serializes and sends a payload object. This works correctly with nested
        dataclasses thanks to `asdict`'s recursive behavior. (No changes here)
        """
        payload_cls = type(payload_instance)
        event_type = CLASS_TO_TYPE_MAP.get(payload_cls)
        
        if event_type is None:
            raise TypeError(f"Object of type {payload_cls.__name__} is not a registered payload.")
            
        payload_dict = asdict(payload_instance)
        message = {"type": event_type, "payload": payload_dict}
        
        json.dump(message, self.output)
        self.output.write('\n')
        self.output.flush()

    def run(self):
        """
        Starts the blocking listener loop. (MODIFIED)
        Uses the `_from_dict` helper for robust deserialization.
        """
        for line in self.input:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            event_type = data['type']
            payload_dict = data['payload']
            
            payload_cls = TYPE_TO_CLASS_MAP[event_type]
            
            # MODIFIED LINE: Use the recursive helper instead of direct instantiation.
            payload_instance = _from_dict(payload_cls, payload_dict)
            
            handler_method_name = f"on_{event_type}"
            handler_method = getattr(self.handler, handler_method_name)
            
            handler_method(payload_instance)
```

## Implementation plan

### **Phase 1: Build the Protocol Foundation**

* Create the `protocol` directory and modules: `registry.py`, `messages.py`, `communicator.py`.
* Implement the `@register` decorator and the forward/reverse lookup maps.
* Define a few representative payload dataclasses in `messages.py`, including one with a nested dataclass.
* Write unit tests to verify the registration system works as expected.

### **Phase 2: Implement the Core `Communicator`**

* Implement the `Communicator` class with its `__init__`, `send`, and `run` methods.
* Include the `_from_dict` private helper function for recursive deserialization.
* Create an example `EventHandler` class for testing purposes.
* Write unit tests for the `Communicator` using mock streams (e.g., `io.StringIO`).

### **Phase 3: Single-Process Integration Test**

* Create a test script to validate the end-to-end flow.
* Instantiate two `Communicator`s linked by an `io.StringIO` object.
* Run the "receiver" in a background thread.
* Have the main thread use the "sender" to send all defined message types.
* Assert that all messages are received and deserialized correctly.

### **Phase 4: Full Two-Process Implementation**

* Create the final `parent_process.py` and `child_process.py` scripts.
* In the parent, use the `subprocess` module to launch the child, capturing its pipes.
* Instantiate and run the `Communicator` in both processes, connected to the appropriate pipes.
* Implement the necessary threading in the parent (UI) and child (trainer) to run the `Communicator` alongside the main application logic.