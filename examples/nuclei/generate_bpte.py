#!/usr/bin/env python3

import torch
from executorch.backends.nuclei.quantizer.quantizer import NucleiQuantizer
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from torchvision.models import mobilenet_v2
from torchvision.models import MobileNet_V2_Weights
from nuclei_example import transformed_img, sample_generator

# Init model
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model = model.eval()

# Create and configure quantizer to use a symmetric quantization config globally on all nodes
quantizer = NucleiQuantizer()

exported_program = torch.export.export(model, (transformed_img,))
graph_module = exported_program.module()

# Post training quantization
quantized_graph_module = prepare_pt2e(graph_module, quantizer)
N_CALIBRATION_SAMPLES=100
for i, (calibration_img, _, _) in enumerate(sample_generator(N_CALIBRATION_SAMPLES)): 
    quantized_graph_module(calibration_img)
    print(f"{i+1}/{N_CALIBRATION_SAMPLES} samples calibrated.", end='\r')

quantized_graph_module = convert_pt2e(quantized_graph_module)

# Create a new exported program using the quantized_graph_module
quantized_exported_program = torch.export.export(quantized_graph_module, (transformed_img,))

import os
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge,
)
from executorch.backends.nuclei.passes.nuclei_pass_manager import NucleiPassManager

# Create compile config for Cortex-M lowering
config = EdgeCompileConfig(
            preserve_ops=[
                torch.ops.aten.linear.default,
                torch.ops.aten.hardsigmoid.default,
                torch.ops.aten.hardsigmoid_.default,
                torch.ops.aten.hardswish.default,
                torch.ops.aten.hardswish_.default,
            ],
            _check_ir_validity=False,
            _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
        )

# Lower the exported program for the Cortex-M backend - note to_edge usage rather than to_edge_transform_and_lower, currently required to use preserve_ops w/o partitioner.
edge_program_manager = to_edge(
            quantized_exported_program,
            compile_config=config,
        )

# Run pass manager on the forward graph_module - use of pass_manager.transform() over edge_program_mangager.transform() is currently required to ensure that the passes can modify the exported_program and not only the graph_module.
pass_manager = NucleiPassManager(edge_program_manager.exported_program())
edge_program_manager._edge_programs["forward"] = pass_manager.transform()

# Serialize edge program
executorch_program_manager = edge_program_manager.to_executorch(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
        )

from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools import BundledProgram
from executorch.devtools.bundled_program.serialize import serialize_from_bundled_program_to_flatbuffer

transformed_img, label, original_img = next(sample_generator())
test_case = MethodTestCase(
    inputs=transformed_img,
    expected_outputs=None)

test_suite = MethodTestSuite(
                method_name="forward",
                test_cases=[test_case],
            )

bundled_program = BundledProgram(executorch_program_manager, [test_suite])
bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(bundled_program)

cwd_dir = os.getcwd()
pte_base_name = "nuclei_example"
pte_name = pte_base_name + ".bpte"
pte_path = os.path.join(cwd_dir, pte_name)
with open(pte_path, "wb") as file:
    file.write(bundled_program_buffer)

