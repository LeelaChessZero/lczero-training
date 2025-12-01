# Description: Evaluation script for comparing model outputs and calculating losses.
#
# This script provides functionalities to evaluate a trained model by processing
# data samples, calculating losses, and comparing outputs against an ONNX model.
# It supports dumping tensors and results to various formats for analysis.

import json
import logging
import math
import shelve
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Optional,
    Sequence,
    TextIO,
    Tuple,
)

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from google.protobuf import text_format

from lczero_training.dataloader import DataLoader, make_dataloader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.state import TrainingState
from proto import data_loader_config_pb2
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)

RELATIVE_DIFFERENCE_EPSILON = 1e-4
HEADS = ("wdl", "policy", "movesleft")


@dataclass
class DiffRecord:
    """Stores information about the difference between JAX and ONNX outputs."""

    batch: int
    sample: int
    index: Tuple[int, ...]
    diff: float
    jax_value: float
    onnx_value: float


def _tensor_to_list(obj: Any) -> Any:
    """Recursively converts JAX/Numpy arrays to Python lists for serialization."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: _tensor_to_list(value) for key, value in obj.items()}
    return obj


# --- Diff statistics helpers ---


def _bin_counts(values: np.ndarray) -> Dict[str, Any]:
    """Counts values into bins based on powers of 2."""
    flat = np.asarray(values).ravel()
    zero_count = int(np.count_nonzero(flat == 0.0))
    non_zero = flat[flat != 0.0]

    bins: Dict[int, int] = {}
    if non_zero.size > 0:
        exponents = np.floor(np.log2(non_zero)).astype(int)
        unique, counts = np.unique(exponents, return_counts=True)
        bins = {int(exp): int(count) for exp, count in zip(unique, counts)}

    return {"zero": zero_count, "bins": bins}


def _format_bound(value: float) -> str:
    """Formats a float for display in statistics."""
    if value >= 1 and math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.6g}"


def _format_stats(stats: Dict[str, Any]) -> str:
    """Formats bin statistics into a readable string."""
    lines = [f"  zero={stats['zero']}"]
    for exponent in sorted(stats["bins"].keys(), reverse=True):
        lower = 2**exponent
        upper = 2 ** (exponent + 1)
        lines.append(
            f"  [{_format_bound(lower)}; {_format_bound(upper)})="
            f"{stats['bins'][exponent]}"
        )
    return "\n".join(lines)


def _collect_diff_statistics(
    jax_output: np.ndarray, onnx_output: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Collects absolute and relative difference statistics."""
    abs_diff = np.abs(jax_output - onnx_output)
    abs_stats = _bin_counts(abs_diff)

    mask = np.abs(jax_output) >= RELATIVE_DIFFERENCE_EPSILON
    if not np.any(mask):
        return abs_diff, abs_stats, None

    rel_diff = np.zeros_like(abs_diff, dtype=np.float64)
    np.divide(abs_diff, np.abs(jax_output), out=rel_diff, where=mask)
    rel_stats = _bin_counts(rel_diff[mask])
    return abs_diff, abs_stats, rel_stats


class Dumper:
    """Handles dumping of evaluation artifacts."""

    def __init__(
        self,
        to_stdout: bool,
        to_file: Optional[str],
        to_shelve: Optional[str],
        to_json: Optional[str],
    ):
        self.to_stdout = to_stdout
        self.shelve_path = to_shelve
        self.json_path = to_json
        self.file_handle: Optional[TextIO] = (
            open(to_file, "w") if to_file else None
        )

    def dump_tensors(self, tensors: dict, prefix: str) -> None:
        """Dumps tensors to stdout or a text file."""
        if not self.to_stdout and not self.file_handle:
            return

        lines = [f"# === {prefix} TENSORS ==="]
        for name, tensor in tensors.items():
            lines.append(f"{name} = {str(_tensor_to_list(tensor))}")
        lines.append("")
        output_text = "\n".join(lines)

        if self.to_stdout:
            print(output_text)
        if self.file_handle:
            self.file_handle.write(output_text)
            self.file_handle.flush()

    def dump_structured(self, batch: dict, outputs: dict, losses: dict) -> None:
        """Dumps results to structured formats like JSON or shelve."""
        if not self.shelve_path and not self.json_path:
            return

        all_data = {
            **_tensor_to_list(batch),
            **_tensor_to_list(outputs),
            **_tensor_to_list(losses),
        }
        key = f"sample-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        if self.shelve_path:
            self._dump_to_shelve(key, all_data)
        if self.json_path:
            self._dump_to_json(key, all_data)

    def _dump_to_shelve(self, key: str, data: dict) -> None:
        assert self.shelve_path is not None
        with shelve.open(self.shelve_path) as db:
            db[key] = data
        logger.info("Dumped data to shelve with key: %s", key)

    def _dump_to_json(self, key: str, data: dict) -> None:
        assert self.json_path is not None
        try:
            with open(self.json_path, "r") as f:
                json_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            json_data = {}
        json_data[key] = data
        with open(self.json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info("Dumped data to JSON with key: %s", key)

    def close(self) -> None:
        if self.file_handle:
            self.file_handle.close()


class OnnxComparator:
    """Handles comparison of JAX model outputs with ONNX model outputs."""

    def __init__(self, onnx_model_path: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for ONNX comparison."
            ) from exc

        self.session = ort.InferenceSession(onnx_model_path)
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError("ONNX model must define at least one input.")
        self.input_name = inputs[0].name
        logger.info("Loaded ONNX model for comparison from %s", onnx_model_path)

        self.head_mapping_logged = False
        self.worst_records: Dict[str, Optional[DiffRecord]] = {
            head: None for head in HEADS
        }
        self.onnx_outputs: Dict[str, jax.Array] = {}

    def compare(
        self,
        jax_outputs: Dict[str, jax.Array],
        onnx_inputs_np: np.ndarray,
        sample_index: int,
    ) -> None:
        """Runs comparison for a single sample."""
        raw_onnx_outputs = self.session.run(
            None, {self.input_name: onnx_inputs_np}
        )
        if len(raw_onnx_outputs) != 3:
            raise ValueError(
                "Expected three outputs (wdl, policy, movesleft) from ONNX model."
            )

        jax_outputs_np = {k: np.asarray(v) for k, v in jax_outputs.items()}
        aligned_onnx, head_indices = self._align_onnx_outputs(
            jax_outputs_np, raw_onnx_outputs
        )

        if not self.head_mapping_logged:
            order = ", ".join(f"{h}=output[{head_indices[h]}]" for h in HEADS)
            logger.info("Aligned ONNX outputs to heads as: %s", order)
            self.head_mapping_logged = True

        self.onnx_outputs = {
            "onnx_value_pred": jnp.asarray(aligned_onnx["wdl"]),
            "onnx_policy_pred": jnp.asarray(aligned_onnx["policy"]),
            "onnx_movesleft_pred": jnp.asarray(aligned_onnx["movesleft"]),
        }

        for head in HEADS:
            record = self._log_diff_stats(
                head, jax_outputs_np[head], aligned_onnx[head], sample_index
            )
            current = self.worst_records[head]
            if current is None or record.diff > current.diff:
                self.worst_records[head] = record

    def log_summary(self) -> None:
        """Logs the worst difference found for each head."""
        for head in HEADS:
            record = self.worst_records[head]
            if record:
                logger.info(
                    "Worst ONNX abs diff for %s head: "
                    "batch=%d sample=%d index=%s diff=%0.6g "
                    "jax=%0.6g onnx=%0.6g",
                    head,
                    record.batch,
                    record.sample,
                    record.index,
                    record.diff,
                    record.jax_value,
                    record.onnx_value,
                )

    def _log_diff_stats(
        self,
        head: str,
        jax_output: np.ndarray,
        onnx_output: np.ndarray,
        sample_index: int,
    ) -> DiffRecord:
        if jax_output.shape != onnx_output.shape:
            raise ValueError(
                f"Shape mismatch for {head} head: "
                f"JAX {jax_output.shape} vs ONNX {onnx_output.shape}."
            )

        abs_diff, abs_stats, rel_stats = _collect_diff_statistics(
            jax_output, onnx_output
        )

        logger.info(
            "Batch %d %s head ONNX abs diff stats:\n%s",
            sample_index,
            head,
            _format_stats(abs_stats),
        )
        if rel_stats:
            logger.info(
                "Batch %d %s head ONNX rel diff stats:\n%s",
                sample_index,
                head,
                _format_stats(rel_stats),
            )
        else:
            logger.info(
                "Batch %d %s head ONNX rel diff stats: skipped (all |jax| < %.1e)",
                sample_index,
                head,
                RELATIVE_DIFFERENCE_EPSILON,
            )

        max_loc = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
        return DiffRecord(
            batch=sample_index,
            sample=int(max_loc[0]) if max_loc else 0,
            index=tuple(int(i) for i in max_loc),
            diff=float(abs_diff[max_loc]),
            jax_value=float(jax_output[max_loc]),
            onnx_value=float(onnx_output[max_loc]),
        )

    def _align_onnx_outputs(
        self,
        jax_outputs: Dict[str, np.ndarray],
        onnx_outputs: Sequence[np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """Matches ONNX outputs to heads regardless of ordering differences."""
        remaining = [(i, np.asarray(o)) for i, o in enumerate(onnx_outputs)]
        aligned: Dict[str, np.ndarray] = {}
        indices: Dict[str, int] = {}

        def pop_match(
            predicate: Callable[[np.ndarray], bool],
        ) -> Optional[Tuple[int, np.ndarray]]:
            for i, candidate in enumerate(remaining):
                if predicate(candidate[1]):
                    return remaining.pop(i)
            return None

        for head in HEADS:
            shape = jax_outputs[head].shape
            match = pop_match(lambda arr: arr.shape == shape)
            if match:
                idx, array = match
                aligned[head], indices[head] = array, idx

        for head in HEADS:
            if head in aligned:
                continue
            size = jax_outputs[head].size
            match = pop_match(lambda arr: arr.size == size)
            if match:
                idx, array = match
                aligned[head], indices[head] = array, idx

        if len(aligned) != len(HEADS):
            rem_shapes = [arr.shape for _, arr in remaining]
            raise ValueError(
                "Could not align ONNX outputs with JAX outputs. "
                f"Aligned: {list(aligned.keys())}; Unmatched shapes: {rem_shapes}"
            )

        for head in HEADS:
            aligned[head] = self._reshape_output(
                aligned[head], jax_outputs[head].shape, head
            )
        return aligned, indices

    def _reshape_output(
        self, array: np.ndarray, target_shape: Tuple[int, ...], head: str
    ) -> np.ndarray:
        if array.shape == target_shape:
            return array
        if array.size != int(np.prod(target_shape)):
            raise ValueError(
                f"Cannot reshape ONNX output for {head}: source shape "
                f"{array.shape}, target shape {target_shape}."
            )
        try:
            return np.reshape(array, target_shape)
        except ValueError as exc:
            raise ValueError(
                f"Failed to reshape ONNX output for {head} to {target_shape}."
            ) from exc


class Evaluation:
    """Orchestrates the model evaluation process."""

    def __init__(self, loss_fn: LczeroLoss):
        self.loss_fn = loss_fn

    def run(
        self,
        model: LczeroModel,
        datagen: Generator[Tuple[np.ndarray, ...], None, None],
        num_samples: int,
        dumper: Dumper,
        onnx_comparator: Optional[OnnxComparator],
        softmax_jax_wdl: bool,
    ) -> None:
        loss_vfn = jax.vmap(self._loss_for_grad, in_axes=(None, 0), out_axes=0)
        model_output_vfn = jax.vmap(
            self._model_for_output, in_axes=(None, 0), out_axes=0
        )

        for i in range(num_samples):
            logger.info("Processing sample %d/%d", i, num_samples)
            self._process_sample(
                model,
                datagen,
                i,
                dumper,
                onnx_comparator,
                loss_vfn,
                model_output_vfn,
                softmax_jax_wdl,
            )
            logger.info("Sample %d complete", i)

    def _process_sample(
        self,
        model: LczeroModel,
        datagen: Generator[Tuple[np.ndarray, ...], None, None],
        sample_idx: int,
        dumper: Dumper,
        onnx_comparator: Optional[OnnxComparator],
        loss_vfn: Callable[
            [LczeroModel, Dict[str, jax.Array]],
            Tuple[jax.Array, Dict[str, jax.Array]],
        ],
        model_output_vfn: Callable[
            [LczeroModel, jax.Array], Tuple[jax.Array, jax.Array, jax.Array]
        ],
        softmax_jax_wdl: bool,
    ) -> None:
        inputs, policy, values, _, movesleft = next(datagen)
        logger.info("Fetched batch from dataloader")

        batch = {
            "inputs": jax.device_put(inputs),
            "value_targets": jax.device_put(values),
            "policy_targets": jax.device_put(policy),
            "movesleft_targets": jax.device_put(movesleft),
        }
        dumper.dump_tensors(batch, "INPUT")

        value_pred, policy_pred, movesleft_pred = model_output_vfn(
            model, batch["inputs"]
        )
        if softmax_jax_wdl:
            value_pred = jax.nn.softmax(value_pred, axis=-1)

        outputs = {
            "value_pred": value_pred,
            "policy_pred": policy_pred,
            "movesleft_pred": movesleft_pred,
        }

        if onnx_comparator:
            jax_outputs_for_onnx = {
                "wdl": value_pred,
                "policy": policy_pred,
                "movesleft": movesleft_pred,
            }
            onnx_inputs_np = np.asarray(inputs).copy()
            onnx_inputs_np[:, 109, ...] *= 99
            onnx_comparator.compare(
                jax_outputs_for_onnx, onnx_inputs_np, sample_idx
            )
            outputs.update(onnx_comparator.onnx_outputs)

        dumper.dump_tensors(outputs, "OUTPUT")

        per_sample_loss, unweighted_losses = loss_vfn(model, batch)
        losses = {
            "per_sample_data_loss": per_sample_loss,
            "unweighted_losses": unweighted_losses,
        }
        dumper.dump_tensors(losses, "LOSSES")
        dumper.dump_structured(batch, outputs, losses)

    def _loss_for_grad(
        self, model_arg: LczeroModel, batch_arg: dict
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        return self.loss_fn(
            model_arg,
            inputs=batch_arg["inputs"],
            value_targets=batch_arg["value_targets"],
            policy_targets=batch_arg["policy_targets"],
            movesleft_targets=batch_arg["movesleft_targets"],
        )

    @staticmethod
    def _model_for_output(
        model_arg: LczeroModel, inputs_arg: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        return model_arg(inputs_arg)


def from_dataloader(
    loader: DataLoader,
) -> Generator[Tuple[np.ndarray, ...], None, None]:
    """Infinetely yields batches from a DataLoader."""
    while True:
        yield loader.get_next()


def _load_model_from_checkpoint(config: RootConfig) -> LczeroModel:
    """Loads a model from the latest checkpoint."""
    if not config.training.checkpoint.path:
        logger.error("Checkpoint path must be set in the configuration.")
        sys.exit(1)

    mgr = ocp.CheckpointManager(
        config.training.checkpoint.path,
        options=ocp.CheckpointManagerOptions(create=True),
    )
    state = TrainingState.new_from_config(config.model, config.training)
    restored_state = mgr.restore(
        mgr.latest_step(), args=ocp.args.PyTreeRestore(state)
    )
    logger.info("Restored checkpoint from %s", config.training.checkpoint.path)

    assert isinstance(restored_state, TrainingState)
    model_graph, _ = nnx.split(
        LczeroModel(config.model, rngs=nnx.Rngs(params=42))
    )
    return nnx.merge(model_graph, restored_state.jit_state.model_state)


def _get_dataloader_config(
    config: RootConfig, batch_size_override: Optional[int]
) -> data_loader_config_pb2.DataLoaderConfig:
    """Gets the dataloader config, overriding batch size if specified."""
    dl_config = config.data_loader
    if batch_size_override is None:
        return dl_config

    for stage in dl_config.stage:
        if stage.HasField("tensor_generator"):
            stage.tensor_generator.batch_size = batch_size_override
            logger.info("Overriding batch size to %d", batch_size_override)
            return dl_config

    raise ValueError(
        "tensor_generator stage is required to override batch size"
    )


def eval(
    config_filename: str,
    num_samples: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    dump_to_stdout: bool = False,
    dump_to_file: Optional[str] = None,
    dump_to_shelve: Optional[str] = None,
    dump_to_json: Optional[str] = None,
    onnx_model: Optional[str] = None,
    softmax_jax_wdl: bool = True,
) -> None:
    """Main function to run the evaluation."""
    jnp.set_printoptions(threshold=sys.maxsize, suppress=False)

    config = RootConfig()
    logger.info("Reading configuration from: %s", config_filename)
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)

    model = _load_model_from_checkpoint(config)
    dl_config = _get_dataloader_config(config, batch_size_override)
    evaluation = Evaluation(loss_fn=LczeroLoss(config=config.training.losses))
    dumper = Dumper(dump_to_stdout, dump_to_file, dump_to_shelve, dump_to_json)
    onnx_comparator = OnnxComparator(onnx_model) if onnx_model else None

    samples_to_process = num_samples if num_samples is not None else 10
    logger.info("Starting evaluation with %d samples", samples_to_process)

    try:
        evaluation.run(
            model=model,
            datagen=from_dataloader(make_dataloader(dl_config)),
            num_samples=samples_to_process,
            dumper=dumper,
            onnx_comparator=onnx_comparator,
            softmax_jax_wdl=softmax_jax_wdl,
        )
    finally:
        dumper.close()
        if onnx_comparator:
            onnx_comparator.log_summary()

    logger.info("Evaluation complete")
