# lc0-weights - Weight Manipulation Tool

## Overview

`lc0-weights` is a command-line tool and Python library for manipulating Leela Chess Zero neural network weight files. It enables arithmetic operations on networks, component grafting, and format conversion without requiring JAX or other heavy dependencies.

**Key capabilities:**

* **Arithmetic operations**: Add, subtract, multiply networks (e.g., model interpolation/averaging)
* **Grafting**: Replace specific network components (policy heads, value heads, encoder layers)
* **Format conversion**: Convert between LINEAR16, FLOAT16, and BFLOAT16 encodings
* **Pure Python/NumPy**: No JAX or TensorFlow dependencies required

## Quick Start

Interpolate two networks with equal weights:

```bash
# Using inline expression
uv run lc0-weights \
  --expr "output = weights('network_a.pb.gz') * 0.5 + weights('network_b.pb.gz') * 0.5" \
  --output interpolated.pb.gz

# Using a script file
echo "output = weights('network_a.pb.gz') * 0.5 + weights('network_b.pb.gz') * 0.5" > interpolate.py
uv run lc0-weights interpolate.py --output interpolated.pb.gz

# Using stdin
echo "output = weights('network_a.pb.gz') * 0.5 + weights('network_b.pb.gz') * 0.5" | \
  uv run lc0-weights --output interpolated.pb.gz
```

## Command-Line Interface

### Arguments

| Argument     | Required | Description                                                          |
| ------------ | -------- | -------------------------------------------------------------------- |
| `--expr`     | No       | Python expression to execute                                         |
| `script`     | No       | Path to Python script file (positional, used if --expr not given)   |
| `--input`    | No       | Pre-load input as `NAME=PATH` (can be used multiple times)          |
| `--output`   | No       | Output path (if `output` variable is set in expression)             |
| `--encoding` | No       | Output encoding format: LINEAR16, FLOAT16 (default), or BFLOAT16    |

**Note:** You must provide either `--expr`, a script file path, or pipe input via stdin.

### Available Functions in --expr

When executing expressions with `--expr`, the following are available:

* `weights(path)`: Load a weight file
* `save(net, path, encoding='FLOAT16')`: Save a network
* `np`: NumPy module
* `lc0`: Protobuf module (for accessing constants)

### CLI Examples

#### Simple Interpolation

**Using inline expression:**
```bash
uv run lc0-weights \
  --expr "output = weights('A.pb.gz') * 0.5 + weights('B.pb.gz') * 0.5" \
  --output result.pb.gz
```

**Using a script file:**
```bash
# Create script file
cat > interpolate.py << 'EOF'
A = weights('A.pb.gz')
B = weights('B.pb.gz')
output = A * 0.5 + B * 0.5
EOF

# Run script
uv run lc0-weights interpolate.py --output result.pb.gz
```

**Using stdin:**
```bash
echo "output = weights('A.pb.gz') * 0.5 + weights('B.pb.gz') * 0.5" | \
  uv run lc0-weights --output result.pb.gz
```

#### Using Input Aliases

Pre-load networks to simplify expressions:

```bash
uv run lc0-weights \
  --input A=network_a.pb.gz \
  --input B=network_b.pb.gz \
  --expr "output = A * 0.9 + B * 0.1" \
  --output result.pb.gz
```

#### Grafting a Policy Head

Replace the policy head of one network with another:

**Using inline expression:**
```bash
uv run lc0-weights --expr "
base = weights('base_network.pb.gz')
donor = weights('network_with_better_policy.pb.gz')
base.weights.policy = donor.weights.policy
base.save('grafted.pb.gz')
"
```

**Using a script file (recommended for multi-line operations):**
```bash
# Create script
cat > graft_policy.py << 'EOF'
base = weights('base_network.pb.gz')
donor = weights('network_with_better_policy.pb.gz')
base.weights.policy = donor.weights.policy
base.save('grafted.pb.gz')
EOF

# Run script
uv run lc0-weights graft_policy.py
```

#### Format Conversion

Convert a network to BFLOAT16 encoding:

```bash
uv run lc0-weights \
  --input net=network.pb.gz \
  --expr "output = net" \
  --output converted.pb.gz \
  --encoding BFLOAT16
```

#### Complex Expression

Weighted average with custom formula:

```bash
uv run lc0-weights \
  --input A=net1.pb.gz \
  --input B=net2.pb.gz \
  --input C=net3.pb.gz \
  --expr "output = A * 0.5 + B * 0.3 + C * 0.2" \
  --output averaged.pb.gz
```

## Python Library Usage

### Importing

```python
from lczero_training.tools import load_weights, save_weights
```

### Loading and Saving Weights

```python
# Load a network
net = load_weights("network.pb.gz")

# Save with different encoding
save_weights(net, "output.pb.gz", encoding="FLOAT16")
```

### Arithmetic Operations

```python
# Load networks
net_a = load_weights("network_a.pb.gz")
net_b = load_weights("network_b.pb.gz")

# Interpolation (model averaging)
interpolated = net_a * 0.7 + net_b * 0.3

# Addition
combined = net_a + net_b

# Subtraction
difference = net_a - net_b

# Scalar multiplication
scaled = net_a * 0.5

# Save result
save_weights(interpolated, "result.pb.gz")
```

### Accessing and Modifying Components

```python
net = load_weights("network.pb.gz")

# Access nested weight arrays
q_weights = net.weights.encoder[0].mha.q_w.value  # Returns NumPy array
print(q_weights.shape)

# Modify weights
net.weights.encoder[0].mha.q_w.value = q_weights * 1.1

# Save modified network
save_weights(net, "modified.pb.gz")
```

### Grafting Components

```python
base = load_weights("base.pb.gz")
donor = load_weights("donor.pb.gz")

# Replace policy head
base.weights.policy = donor.weights.policy

# Replace value head
base.weights.value_heads = donor.weights.value_heads

# Replace specific encoder layer
base.weights.encoder[0] = donor.weights.encoder[0]

save_weights(base, "grafted.pb.gz")
```

## Weight Encoding Formats

The tool supports three encoding formats for weight storage:

| Format   | Description                                                                      | Precision | File Size |
| -------- | -------------------------------------------------------------------------------- | --------- | --------- |
| LINEAR16 | Quantized 16-bit integer with min/max range. Default for Lc0.                   | ~4 digits | Smallest  |
| FLOAT16  | Native IEEE 754 half-precision floating point.                                   | ~3 digits | Medium    |
| BFLOAT16 | Brain float 16 (truncated float32). Better range than FLOAT16, less precision.  | ~2 digits | Medium    |

**Notes:**

* LINEAR16 provides good compression with acceptable precision for neural networks
* FLOAT16 is the default for this tool (good balance of precision and size)
* BFLOAT16 is useful when training range matters more than mantissa precision
* All formats are converted to float32 when loaded for arithmetic operations

## Common Use Cases

### Network Interpolation (Model Averaging)

Combine two networks to create a smoother model or blend different training runs:

```python
from lczero_training.tools import load_weights, save_weights

net1 = load_weights("run1_final.pb.gz")
net2 = load_weights("run2_final.pb.gz")

# Average the networks
averaged = net1 * 0.5 + net2 * 0.5

save_weights(averaged, "averaged_network.pb.gz")
```

### Exponential Moving Average (EMA)

Update a running average network with a new checkpoint:

```python
ema_net = load_weights("ema.pb.gz")
new_net = load_weights("latest_checkpoint.pb.gz")

# EMA with decay 0.999
ema_updated = ema_net * 0.999 + new_net * 0.001

save_weights(ema_updated, "ema.pb.gz")
```

### Policy Head Replacement

Replace a network's policy head (useful for policy distillation):

```python
student = load_weights("student_network.pb.gz")
teacher = load_weights("teacher_network.pb.gz")

# Replace student's policy head with teacher's
student.weights.policy_heads = teacher.weights.policy_heads

save_weights(student, "student_with_teacher_policy.pb.gz")
```

### Extracting Network Statistics

```python
net = load_weights("network.pb.gz")

# Get statistics from first encoder layer
layer = net.weights.encoder[0].mha.q_w.value
print(f"Shape: {layer.shape}")
print(f"Mean: {layer.mean():.6f}")
print(f"Std: {layer.std():.6f}")
print(f"Min: {layer.min():.6f}")
print(f"Max: {layer.max():.6f}")
```

### Format Conversion for Size Optimization

```python
from lczero_training.tools import load_weights, save_weights

# Load network (any format)
net = load_weights("large_network.pb.gz")

# Save with more aggressive compression
save_weights(net, "compressed_network.pb.gz", encoding="LINEAR16")
```

## Advanced Usage

### Accessing Nested Structures

The weight wrapper provides Pythonic access to the nested protobuf structure:

```python
net = load_weights("network.pb.gz")

# Access input embedding weights
input_weights = net.weights.input.weights.value

# Access specific encoder layer
encoder_layer_5 = net.weights.encoder[5]

# Access multi-head attention components
q_weights = net.weights.encoder[0].mha.q_w.value
k_weights = net.weights.encoder[0].mha.k_w.value
v_weights = net.weights.encoder[0].mha.v_w.value

# Access policy head
policy_weights = net.weights.policy_heads.vanilla.ip_pol_w.value
```

### Complex Weighted Combinations

```python
nets = [load_weights(f"checkpoint_{i}.pb.gz") for i in range(5)]
weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # More weight to recent checkpoints

result = sum(net * w for net, w in zip(nets, weights))
save_weights(result, "weighted_ensemble.pb.gz")
```

### Selective Component Grafting

```python
base = load_weights("base.pb.gz")
donor = load_weights("donor.pb.gz")

# Replace only the first 10 encoder layers
for i in range(10):
    base.weights.encoder[i] = donor.weights.encoder[i]

save_weights(base, "partial_graft.pb.gz")
```

### Working with NumPy Arrays Directly

All weight access returns NumPy arrays, allowing arbitrary transformations:

```python
net = load_weights("network.pb.gz")

# Get layer weights
layer_weights = net.weights.encoder[0].mha.q_w.value

# Apply custom transformation
import numpy as np
layer_weights_normalized = layer_weights / np.linalg.norm(layer_weights, axis=-1, keepdims=True)

# Write back
net.weights.encoder[0].mha.q_w.value = layer_weights_normalized

save_weights(net, "normalized.pb.gz")
```

## Implementation Details

### Lazy Loading

Weights are decoded from their compressed format only when accessed, and cached for subsequent use. This makes the tool memory-efficient when working with large networks.

### File Format Support

Both `.pb` (uncompressed protobuf) and `.pb.gz` (gzip-compressed) formats are supported. The tool automatically detects the format based on the file extension.

### Arithmetic Semantics

* Operations are element-wise across all matching layers
* Networks must have compatible structures (same number of encoder layers, etc.)
* Results are computed in float32 precision regardless of input encoding
