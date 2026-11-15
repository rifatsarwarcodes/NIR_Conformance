# norse 1.1.0: CubaLIF synaptic current leaks towards v_leak, so a neuron with zero input fires
import nir, numpy as np, torch
from norse.torch import from_nir
f, n = np.float32, 1
cuba = nir.CubaLIF(tau_syn=np.full(n, 10, f), tau_mem=np.full(n, 20, f), r=np.ones(n, f), v_leak=np.full(n, 0.6, f),
                   v_threshold=np.ones(n, f), w_in=np.ones(n, f))
g = nir.NIRGraph(nodes={"input": nir.Input(np.array([n])), "lif": cuba, "output": nir.Output(np.array([n]))},
                 edges=[("input", "lif"), ("lif", "output")])
net, state, spikes = from_nir(g, dt=1.0), None, 0
for t in range(200):
    out, state = net(torch.zeros(1, n), state)
    spikes += int(out.sum())
print("spikes with zero input:", spikes, "(expected 0: v settles at v_leak = 0.6 < threshold 1)")
