#!/usr/bin/python3
import torch
import torch.nn as nn
import numpy as np
import os

class WhereModel(nn.Module):
    def forward(self, cond, x, y):
        return torch.where(cond, x, y)

def generate():
    model = WhereModel()
    # (2, 2) shape
    cond = torch.tensor([[True, False], [False, True]])
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])

    onnx_path = "src/SOFIE_core/test/input_models/Where.onnx"
    torch.onnx.export(model, (cond, x, y), onnx_path, 
                      input_names=["cond", "x", "y"], 
                      output_names=["out"],
                      opset_version=9)

    out = torch.where(cond, x, y)
    
    # Generate C++ header
    header_path = "src/SOFIE_core/test/input_models/references/Where.ref.hxx"
    with open(header_path, "w") as f:
        f.write("#ifndef SOFIE_STEST_WHERE_REF\n")
        f.write("#define SOFIE_STEST_WHERE_REF\n\n")
        f.write("namespace Where_ExpectedOutput {\n")
        
        # Write output
        out_flat = out.flatten().numpy()
        f.write("  float output[] = { " + ", ".join([str(v) for v in out_flat]) + " };\n")
        
        # Write condition (uint8 for Alpaka)
        cond_flat = cond.flatten().numpy().astype(np.uint8)
        f.write("  uint8_t cond[] = { " + ", ".join([str(v) for v in cond_flat]) + " };\n")

        # Write x and y
        x_flat = x.flatten().numpy()
        f.write("  float x[] = { " + ", ".join([str(v) for v in x_flat]) + " };\n")
        y_flat = y.flatten().numpy()
        f.write("  float y[] = { " + ", ".join([str(v) for v in y_flat]) + " };\n")
        
        f.write("}\n\n")
        f.write("#endif\n")

if __name__ == "__main__":
    generate()
