<!-- DRAFTS for neuromorphs/NIR — not posted -->

# 1. `Conv1d`/`Conv2d.input_type` ignores `groups`

`Conv2d.__post_init__` sets the input channels to `weight.shape[1]` (`nir/ir/conv.py:124`; `Conv1d` at `:57`). For a grouped convolution `weight.shape[1]` is `C_in / groups`, so a valid grouped conv fails graph type inference.

```python
# upstream/repros/nir_grouped_conv.py  (nir 1.0.8)
w = np.zeros((4, 2, 3, 3), np.float32)          # groups=2  =>  C_in = 4
conv = nir.Conv2d(input_shape=(7, 7), weight=w, stride=1, padding=1, dilation=1, groups=2, bias=np.zeros(4, np.float32))
conv.input_type["input"]                          # [2 7 7], expected [4 7 7]
nir.NIRGraph(nodes={"input": nir.Input(np.array([4, 7, 7])), "conv": conv, "output": nir.Output(np.array([4, 7, 7]))},
             edges=[("input", "conv"), ("conv", "output")])
# ValueError: Type inference error: type mismatch: input.output: [[4 7 7]] -> conv.input: [[2 7 7]]
```

**Suggested fix:** `self.weight.shape[1] * self.groups`, in both classes.

# 2. Files from nir ≥ 1.0.6 cannot be read by nir ≤ 1.0.4 (always-written `v_reset`)

**What happens:**
- From 1.0.6, `LIF`, `IF` and `CubaLIF` gained `v_reset`.
- `nir.write` always serialises it, even when the user never set it (it defaults to zeros).
- nir 1.0.4's `from_dict` rejects the key: `TypeError: LIF.__init__() got an unexpected keyword argument 'v_reset'`.
- `CubaLI` (new in 1.0.5) is likewise unknown to 1.0.4 (`AssertionError` in `str2NIRNode`).

**Why it matters:** `sinabs` 3.1.3 requires `nir<=1.0.4`, and so does `rockpool[nir]` 3.1.0. Those two frameworks cannot read any spiking-neuron graph exported by current snnTorch, Norse or Spyx. The file's `version` field says `1.0.8`, but readers do not check it.

**Possible fixes:**
- (a) Omit `v_reset` on write when it is all zeros, which keeps old readers working for the common case.
- (b) Have readers ignore unknown keys with a warning.
- (c) Have readers check the `version` attribute and give a clear error.

(Either (a) or (b) would have avoided this.) Separately, the Sinabs and Rockpool pins need lifting.

(Verified with nir 1.0.4, 1.0.5, 1.0.6, 1.0.7 and 1.0.8: `v_reset` appears in 1.0.6.)
