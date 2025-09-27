import json
import logging
import shelve
import sys
from datetime import datetime
from typing import Any, Dict, Generator, Optional, TextIO, Tuple

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
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


def from_dataloader(
    loader: DataLoader,
) -> Generator[Tuple[np.ndarray, ...], None, None]:
    while True:
        yield loader.get_next()


class Evaluation:
    def __init__(self, loss_fn: LczeroLoss):
        self.loss_fn = loss_fn

    def run(
        self,
        model: LczeroModel,
        datagen: Generator[Tuple[np.ndarray, ...], None, None],
        num_samples: int,
        dump_to_stdout: bool = False,
        dump_to_file: Optional[str] = None,
        dump_to_shelve: Optional[str] = None,
        dump_to_json: Optional[str] = None,
    ) -> None:
        dump_file: Optional[TextIO] = None
        if dump_to_file:
            dump_file = open(dump_to_file, "w")

        logger.info(f"Starting evaluation with {num_samples} samples")

        def loss_for_grad(
            model_arg: LczeroModel, batch_arg: dict
        ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
            return self.loss_fn(
                model_arg,
                inputs=batch_arg["inputs"],
                value_targets=batch_arg["value_targets"],
                policy_targets=batch_arg["policy_targets"],
                movesleft_targets=batch_arg["movesleft_targets"],
            )

        loss_vfn = jax.vmap(
            loss_for_grad,
            in_axes=(None, 0),
            out_axes=0,
        )

        def model_for_output(
            model_arg: LczeroModel, inputs_arg: jax.Array
        ) -> Tuple[jax.Array, jax.Array, jax.Array]:
            return model_arg(inputs_arg)

        model_output_vfn = jax.vmap(
            model_for_output,
            in_axes=(None, 0),
            out_axes=0,
        )

        try:
            for sample_idx in range(num_samples):
                logger.info(f"Processing sample {sample_idx + 1}/{num_samples}")

                batch = next(datagen)
                b_inputs, b_policy, b_values, _, b_movesleft = batch
                logger.info("Fetched batch from dataloader")

                # Convert numpy arrays to JAX arrays
                b_inputs = jax.device_put(b_inputs)
                b_policy = jax.device_put(b_policy)
                b_values = jax.device_put(b_values)
                b_movesleft = jax.device_put(b_movesleft)

                batch_dict = {
                    "inputs": b_inputs,
                    "value_targets": b_values,
                    "policy_targets": b_policy,
                    "movesleft_targets": b_movesleft,
                }

                if dump_to_stdout or dump_to_file:
                    logger.info("Dumping input tensors")
                    self._dump_tensors(
                        batch_dict, dump_to_stdout, dump_file, "INPUT"
                    )

                value_pred, policy_pred, movesleft_pred = model_output_vfn(
                    model, b_inputs
                )
                outputs = {
                    "value_pred": value_pred,
                    "policy_pred": policy_pred,
                    "movesleft_pred": movesleft_pred,
                }

                if dump_to_stdout or dump_to_file:
                    logger.info("Dumping output tensors")
                    self._dump_tensors(
                        outputs, dump_to_stdout, dump_file, "OUTPUT"
                    )

                per_sample_data_loss, unweighted_losses = loss_vfn(
                    model, batch_dict
                )

                losses_dict = {
                    "per_sample_data_loss": per_sample_data_loss,
                    "unweighted_losses": unweighted_losses,
                }

                if dump_to_stdout or dump_to_file:
                    logger.info("Dumping loss values")
                    self._dump_tensors(
                        losses_dict, dump_to_stdout, dump_file, "LOSSES"
                    )

                if dump_to_shelve:
                    logger.info(
                        f"Dumping to shelve database at {dump_to_shelve}"
                    )
                    self._dump_to_shelve(
                        dump_to_shelve, batch_dict, outputs, losses_dict
                    )

                if dump_to_json:
                    logger.info(f"Dumping to JSON file at {dump_to_json}")
                    self._dump_to_json(
                        dump_to_json, batch_dict, outputs, losses_dict
                    )

                logger.info(f"Sample {sample_idx + 1} complete")

        finally:
            if dump_file:
                dump_file.close()

        logger.info("Evaluation complete")

    def _tensor_to_list(self, obj: Any) -> Any:
        """Recursively convert tensors to Python lists."""
        if hasattr(obj, "tolist"):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: self._tensor_to_list(value) for key, value in obj.items()
            }
        else:
            return obj

    def _dump_to_shelve(
        self,
        shelve_path: str,
        batch_dict: dict,
        outputs: dict,
        losses_dict: dict,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        key = f"sample-{timestamp}"

        # Combine all data into a single dictionary
        all_data = {}
        all_data.update(
            {k: self._tensor_to_list(v) for k, v in batch_dict.items()}
        )
        all_data.update(
            {k: self._tensor_to_list(v) for k, v in outputs.items()}
        )
        all_data.update(
            {k: self._tensor_to_list(v) for k, v in losses_dict.items()}
        )

        with shelve.open(shelve_path) as db:
            db[key] = all_data
        logger.info(f"Dumped data to shelve with key: {key}")

    def _dump_to_json(
        self,
        json_path: str,
        batch_dict: dict,
        outputs: dict,
        losses_dict: dict,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        key = f"sample-{timestamp}"

        # Combine all data into a single dictionary
        all_data = {}
        all_data.update(
            {k: self._tensor_to_list(v) for k, v in batch_dict.items()}
        )
        all_data.update(
            {k: self._tensor_to_list(v) for k, v in outputs.items()}
        )
        all_data.update(
            {k: self._tensor_to_list(v) for k, v in losses_dict.items()}
        )

        # Load existing data or create new structure
        try:
            with open(json_path, "r") as f:
                json_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            json_data = {}

        json_data[key] = all_data

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Dumped data to JSON with key: {key}")

    def _dump_tensors(
        self,
        tensors: dict,
        dump_to_stdout: bool,
        dump_file: Optional[TextIO],
        prefix: str,
    ) -> None:
        output_lines = []
        output_lines.append(f"# === {prefix} TENSORS ===")
        for name, tensor in tensors.items():
            output_lines.append(f"{name} = {str(self._tensor_to_list(tensor))}")
        output_lines.append("")

        output_text = "\n".join(output_lines)

        if dump_to_stdout:
            print(output_text)

        if dump_file:
            dump_file.write(output_text)
            dump_file.flush()


def eval(
    config_filename: str,
    num_samples: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    dump_to_stdout: bool = False,
    dump_to_file: Optional[str] = None,
    dump_to_shelve: Optional[str] = None,
    dump_to_json: Optional[str] = None,
) -> None:
    # Set JAX numpy print options to show full tensors
    jnp.set_printoptions(threshold=sys.maxsize, suppress=False)

    config = RootConfig()
    logger.info("Reading configuration from proto file")
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)

    if config.training.checkpoint.path is None:
        logger.error("Checkpoint path must be set in the configuration.")
        sys.exit(1)

    checkpoint_mgr = ocp.CheckpointManager(
        config.training.checkpoint.path,
        options=ocp.CheckpointManagerOptions(
            create=True,
        ),
    )

    logger.info("Creating state from configuration")
    empty_state = TrainingState.new_from_config(
        model_config=config.model,
        training_config=config.training,
    )
    logger.info("Restoring checkpoint")
    training_state = checkpoint_mgr.restore(
        None, args=ocp.args.PyTreeRestore(empty_state)
    )
    logger.info("Restored checkpoint")

    assert isinstance(training_state, TrainingState)
    model_graphdef, _ = nnx.split(
        LczeroModel(config=config.model, rngs=nnx.Rngs(params=42))
    )
    model = nnx.merge(model_graphdef, training_state.jit_state.model_state)

    dataloader_config = config.data_loader
    if batch_size_override is not None:
        tensor_stage_config = None
        for stage in dataloader_config.stage:
            if stage.HasField("tensor_generator"):
                tensor_stage_config = stage.tensor_generator
                break

        if tensor_stage_config is None:
            raise ValueError(
                "tensor_generator stage is required to override batch size"
            )

        tensor_stage_config.batch_size = batch_size_override
        logger.info(f"Overriding batch size to {batch_size_override}")

    evaluation = Evaluation(loss_fn=LczeroLoss(config=config.training.losses))

    samples_to_process = num_samples if num_samples is not None else 10
    logger.info(f"Starting evaluation with {samples_to_process} samples")

    evaluation.run(
        model=model,
        datagen=from_dataloader(make_dataloader(dataloader_config)),
        num_samples=samples_to_process,
        dump_to_stdout=dump_to_stdout,
        dump_to_file=dump_to_file,
        dump_to_shelve=dump_to_shelve,
        dump_to_json=dump_to_json,
    )
