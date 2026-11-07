"""Export DepthForge-Net to ONNX so the deployed app can run without PyTorch.

  python scripts/export_onnx.py        # -> weights/depthforge_net.onnx

onnxruntime is ~15 MB vs ~800 MB for PyTorch, which lets the app fit free hosting
tiers (e.g. Render free: 512 MB RAM). Outputs match PyTorch to ~1e-5.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from depthforge.model import DepthForgeNet  # noqa: E402


def main():
    ck = torch.load(ROOT / "weights" / "depthforge_net.pt", map_location="cpu", weights_only=False)
    m = DepthForgeNet(ck.get("width", 32)); m.load_state_dict(ck["state_dict"]); m.eval()
    T = int(ck.get("crop", 128))
    d, r = torch.rand(2, 1, T, T), torch.rand(2, 3, T, T)
    out = ROOT / "weights" / "depthforge_net.onnx"
    torch.onnx.export(m, (d, r), str(out), input_names=["depth", "rgb"], output_names=["out"],
                      dynamic_axes={"depth": {0: "n"}, "rgb": {0: "n"}, "out": {0: "n"}},
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    d2, r2 = torch.rand(5, 1, T, T), torch.rand(5, 3, T, T)
    with torch.no_grad():
        ref = m(d2, r2).numpy()
    got = s.run(None, {"depth": d2.numpy(), "rgb": r2.numpy()})[0]
    print(f"wrote {out} ({out.stat().st_size/2**20:.2f} MB), tile {T}, max |diff| = {np.abs(ref-got).max():.2e}")


if __name__ == "__main__":
    main()
