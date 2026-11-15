# nir: Conv2d.input_type ignores `groups` (nir 1.0.8)
import nir, numpy as np
w = np.zeros((4, 2, 3, 3), np.float32)  # C_out=4, C_in/groups=2, groups=2  =>  C_in = 4
conv = nir.Conv2d(input_shape=(7, 7), weight=w, stride=1, padding=1, dilation=1, groups=2, bias=np.zeros(4, np.float32))
print("input_type:", conv.input_type["input"])  # [2 7 7], expected [4 7 7]
nir.NIRGraph(nodes={"input": nir.Input(np.array([4, 7, 7])), "conv": conv, "output": nir.Output(np.array([4, 7, 7]))},
             edges=[("input", "conv"), ("conv", "output")])  # ValueError: Type inference error: type mismatch
