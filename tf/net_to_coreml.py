#!/usr/bin/env python3
import os
from net_to_model import convert
import coremltools as ct

if __name__ == "__main__":
    ##############
    # NET TO MODEL
    args, root_dir, tfp = convert(include_attn_wts_output=False, rescale_rule50=False)

    #################
    # MODEL TO COREML
    input_shape = ct.Shape(shape=(1, 112, 8, 8))

    # Set the compute precision
    compute_precision = ct.precision.FLOAT16
    # compute_precision = ct.precision.FLOAT32

    # Convert the model to CoreML
    coreml_model = ct.convert(
        tfp.model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=input_shape, name="input_1")],
        compute_precision=compute_precision,
    )

    # Get the protobuf spec
    spec = coreml_model._spec

    # Rename the input
    ct.utils.rename_feature(spec, "input_1", "input_planes")

    # Get input names
    input_names = [input.name for input in spec.description.input]

    # Print the input names
    print(f"Renamed input: {input_names}")

    # Set output names
    output_names = ["output_policy", "output_value"]

    if tfp.moves_left:
        output_names.append("output_moves_left")

    # Rename output names
    for i, name in enumerate(output_names):
        # Rename the output
        ct.utils.rename_feature(spec, spec.description.output[i].name, name)

    # Print the output names
    print(f"Renamed output: {[output_i.name for output_i in spec.description.output]}")

    # Set model description
    coreml_model.short_description = f"Lc0 converted from {args.net}"

    # Rebuild the model with the updated spec
    print(f"Rebuilding model with updated spec ...")
    rebuilt_mlmodel = ct.models.MLModel(
        coreml_model._spec, weights_dir=coreml_model._weights_dir
    )

    # Save the CoreML model
    print(f"Saving model ...")
    coreml_model_path = os.path.join(root_dir, f"{args.net}.mlpackage")
    coreml_model.save(coreml_model_path)

    print(f"CoreML model saved at {coreml_model_path}")
