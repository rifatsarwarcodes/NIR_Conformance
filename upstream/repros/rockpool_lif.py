# rockpool 3.1.0 (nir 1.0.4): every nir.LIF import fails
import nir, numpy as np
from rockpool.nn.modules.torch.nir import from_nir
f = np.float32
g = nir.NIRGraph(nodes={"input": nir.Input(np.array([2])), "fc": nir.Affine(np.eye(2, dtype=f), np.zeros(2, f)),
                        "lif": nir.LIF(tau=np.full(2, 0.02, f), r=np.ones(2, f), v_leak=np.zeros(2, f), v_threshold=np.ones(2, f)),
                        "output": nir.Output(np.array([2]))}, edges=[("input", "fc"), ("fc", "lif"), ("lif", "output")])
from_nir(g)  # AttributeError: 'LIF' object has no attribute 'tau_mem'
