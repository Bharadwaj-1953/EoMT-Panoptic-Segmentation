"""Environment checks only: no model construction, downloads, or dataset access."""
import argparse
import importlib
import platform
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--expect-cpu", action="store_true")
parser.add_argument("--require-cuda", action="store_true")
args = parser.parse_args()

import torch
import torchvision
from torchvision.ops import nms

assert sys.version_info[:3] == (3, 11, 12), sys.version
assert torch.__version__.split("+")[0] == "2.7.0", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.22.0", torchvision.__version__
print(f"Python={platform.python_version()} machine={platform.machine()}")
print(f"torch={torch.__version__} torchvision={torchvision.__version__}")
print(f"CUDA build={torch.version.cuda} available={torch.cuda.is_available()}")
if args.expect_cpu:
    assert torch.version.cuda is None, "CPU image unexpectedly contains CUDA PyTorch"

# A small CPU operation and compiled torchvision operator catch mismatched wheels.
assert torch.equal(torch.ones(2, 2) @ torch.ones(2, 2), torch.full((2, 2), 2.0))
boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]])
assert nms(boxes, torch.tensor([0.9, 0.8]), 0.5).tolist() == [0]

source = Path("/workspace/upstream/eomt")
for package in ("models", "datasets", "training"):
    for path in sorted((source / package).glob("*.py")):
        name = package if path.stem == "__init__" else f"{package}.{path.stem}"
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(source), module.__file__
        print(f"import OK: {name}")
importlib.import_module("main")
print("import OK: main (CLI not executed)")

if args.require_cuda:
    assert torch.cuda.is_available(), "NVIDIA CUDA device unavailable"
    value = torch.ones(2, 2, device="cuda") @ torch.ones(2, 2, device="cuda")
    assert value.cpu().tolist() == [[2.0, 2.0], [2.0, 2.0]]
    torch.cuda.synchronize()
    print(f"CUDA operation OK: {torch.cuda.get_device_name(0)}")
print("Environment smoke tests passed; no model or dataset was loaded.")

