# snntorch 1.0.0: nir.IF becomes a leaky neuron; LIF dt hardcoded to 1e-4
import nir, numpy as np
from snntorch.import_nir import import_from_nir
f = np.float32
def graph(neuron):
    return nir.NIRGraph(nodes={"input": nir.Input(np.array([2])), "fc": nir.Linear(np.eye(2, dtype=f)), "n": neuron,
                               "output": nir.Output(np.array([2]))}, edges=[("input", "fc"), ("fc", "n"), ("n", "output")])
m = [m for m in import_from_nir(graph(nir.IF(r=np.ones(2, f), v_threshold=np.ones(2, f)))).modules() if type(m).__name__ == "Leaky"][0]
print("IF ->", type(m).__name__, "beta", m.beta.item() if m.beta.numel() == 1 else m.beta, "reset", m.reset_mechanism)  # beta 0.9, subtract
lif = nir.LIF(tau=np.full(2, 20.0, f), r=np.ones(2, f), v_leak=np.zeros(2, f), v_threshold=np.ones(2, f))  # tau = 20 steps (dt = 1)
m = [m for m in import_from_nir(graph(lif)).modules() if type(m).__name__ == "Leaky"][0]
print("LIF tau=20 -> beta", m.beta.flatten()[0].item(), "(0.95 expected for dt=1; import_from_nir has no dt argument)")
