# nirtorch.load (legacy GraphExecutor): fan-in graph returns a branch's Linear output (via norse 1.1.0)
import nir, numpy as np, torch
from norse.torch import from_nir
rng = np.random.default_rng(0)
f = np.float32
lif = nir.LIF(tau=np.full(4, 20, f), r=np.ones(4, f), v_leak=np.zeros(4, f), v_threshold=np.ones(4, f))
g = nir.NIRGraph(
    nodes={"input": nir.Input(np.array([3])), "fc_a": nir.Linear(rng.normal(size=(4, 3)).astype(f)),
           "fc_b": nir.Linear(rng.normal(size=(4, 3)).astype(f)), "lif": lif, "output": nir.Output(np.array([4]))},
    edges=[("input", "fc_a"), ("input", "fc_b"), ("fc_a", "lif"), ("fc_b", "lif"), ("lif", "output")])
net = from_nir(g, dt=1.0)
print("execution order:", [n.name for n in net.execution_order])  # fc_b runs after output
x = torch.ones(1, 3)
out, _ = net(x)
print("output:", out)                                  # not spikes: real-valued
print("equals fc_b(x):", torch.allclose(out, x @ torch.tensor(g.nodes["fc_b"].weight).T))  # True
