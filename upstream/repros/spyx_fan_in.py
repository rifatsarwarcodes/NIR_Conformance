# spyx 1.0.0: second branch of a fan-in graph is silently dropped
import nir, numpy as np, jax.numpy as jnp, spyx.nir
rng = np.random.default_rng(0)
f = np.float32
def graph(w_b):
    lif = nir.LIF(tau=np.full(4, 20, f), r=np.ones(4, f), v_leak=np.zeros(4, f), v_threshold=np.ones(4, f))
    return nir.NIRGraph(
        nodes={"input": nir.Input(np.array([3])), "fc_a": nir.Linear(np.full((4, 3), 0.5, f)), "fc_b": nir.Linear(w_b),
               "lif": lif, "output": nir.Output(np.array([4]))},
        edges=[("input", "fc_a"), ("input", "fc_b"), ("fc_a", "lif"), ("fc_b", "lif"), ("lif", "output")])
x = jnp.ones((20, 1, 3))
_, y1 = spyx.nir.from_nir(graph(np.full((4, 3), 5.0, f)), x, dt=1.0)
_, y2 = spyx.nir.from_nir(graph(np.zeros((4, 3), f)), x, dt=1.0)
print("spikes with strong branch b:", int(y1.sum()), "| with branch b zeroed:", int(y2.sum()), "(identical => branch b ignored)")
