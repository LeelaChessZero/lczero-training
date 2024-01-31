#!/usr/bin/env python3
import argparse
import os
import yaml
import tfprocess
import coremltools as ct

if __name__ == "__main__":
    ##############
    # NET TO MODEL
    argparser = argparse.ArgumentParser(description="Convert net to coreml.")
    argparser.add_argument(
        "net", type=str, help="Net file to be converted to a model checkpoint."
    )
    argparser.add_argument(
        "--start", type=int, default=0, help="Offset to set global_step to."
    )
    argparser.add_argument(
        "--cfg",
        type=argparse.FileType("r"),
        help="yaml configuration with training parameters",
    )
    argparser.add_argument(
        "-e",
        "--ignore-errors",
        action="store_true",
        help="Ignore missing and wrong sized values.",
    )
    args = argparser.parse_args()
    cfg = yaml.safe_load(args.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))
    START_FROM = args.start

    tfp = tfprocess.TFProcess(cfg)
    tfp.init_net(include_attn_wts_output=False)
    tfp.replace_weights(args.net, args.ignore_errors)
    tfp.global_step.assign(START_FROM)

    root_dir = os.path.join(cfg["training"]["path"], cfg["name"])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    tfp.manager.save(checkpoint_number=START_FROM)
    print("Wrote model to {}".format(tfp.manager.latest_checkpoint))

    #################
    # MODEL TO COREML
    input_shape = ct.Shape(shape=(1, 112, 8, 8))

    # Convert the model to CoreML
    coreml_model = ct.convert(
        tfp.model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=input_shape, name="input_1")],
    )

    # Get the protobuf spec
    spec = coreml_model._spec

    # Rename the input
    ct.utils.rename_feature(spec, "input_1", "input_planes")

    # Get input names
    input_names = [input.name for input in spec.description.input]

    # Print the input names
    print(f"Input names: {input_names}")

    # Set output names
    output_names = ["output_policy", "output_value"]

    if tfp.moves_left:
        output_names.append("output_moves_left")

    # Rename output names
    for i, name in enumerate(output_names):
        # Rename the output
        ct.utils.rename_feature(spec, spec.description.output[i].name, name)

    # Print the output names
    print(f"Output names: {[output_i.name for output_i in spec.description.output]}")

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
