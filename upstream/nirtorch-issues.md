<!-- DRAFT for neuromorphs/NIRTorch — not posted. One issue with three parts, or split into three. -->

# Legacy `nirtorch.load` / `GraphExecutor` returns wrong results silently (fan-in, unsupported nodes, hash-seed dependence)

`nirtorch.load` is deprecated in favour of `nir_to_torch`, but Norse 1.1.0, Sinabs 3.1.3 and Rockpool 3.1.0 still use it, so their users get these behaviours today. Seen with nirtorch 2.6 and 2.0.2.

**1. Fan-in graphs return the wrong node.** For `input → fc_a → lif`, `input → fc_b → lif`, `lif → output`:
- The execution order is `['input', 'fc_a', 'lif', 'output', 'fc_b']`. `fc_b` runs after `lif` and after `output`.
- `forward` then returns `execution_order[-2]` when the last node is a dummy `Output` (`graph_executor.py:166-167`). So the model returns `fc_b(x)`, a real-valued Linear output, instead of the LIF spikes.
- Only the first node in the order receives `data` (`:160`).

```python
# upstream/repros/nirtorch_fan_in.py  (via norse.torch.from_nir)
execution order: ['input', 'fc_a', 'lif', 'output', 'fc_b']
output: tensor([[-3.7897, -1.5928,  1.3256,  1.0528]])
equals fc_b(x): True
```

`nir_to_torch` (the path snnTorch uses) gets this graph right.

**2. Unmappable nodes are skipped without error.** If the user's mapping returns `None` for a node, the node is left out of the execution order (`if node.elem is None: continue`, `:153`). The model then runs and gives a wrong answer:
- A Linear → LI graph imported with Norse has execution order `['input', 'fc']`, and its output is just the Linear output.
- In Norse this silently drops LI, I, CubaLI and AvgPool2d: each output equals the graph with that node removed.
- Graphs with Delay or Scale go through the same skip, but their outcome varies (point 3).

Suggestion: raise, or at least warn, naming the node.

**3. With a skipped node, the outcome is not deterministic.** Importing and running the same file 30 times in one process (`PYTHONHASHSEED` fixed) gives:

| Fixture | Raises | Runs (wrong output) |
|---|---|---|
| Norse `Linear → Scale → LIF` | 13 | 17 |
| Norse `Linear → Delay → LIF` | 12 | 18 |
| Norse `Linear → LIF → Linear → LIF` (control) | 0 | 30, identical every time |

The counts change from run to run. Under nirtorch 2.0.2, Sinabs `Flatten → Linear → LIF` raised in 5 of 30. The likely cause is that `graph.py:79-84` builds `set(graph.node_list)` and `set(node.outgoing_nodes)` of node objects. Those sets are ordered by object identity (memory address), so the order in which the graph is re-wired around a missing node changes from run to run.
