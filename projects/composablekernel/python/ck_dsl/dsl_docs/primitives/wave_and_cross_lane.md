# Wave And Cross-Lane Primitives

AMDGPU's wave64 cross-lane operations are first-class in `ck_dsl`. This page covers them in depth because they recur in attention, reductions, and the SGPR/VGPR balance of any hot K-loop.

Source: `core/ir.py` (`IRBuilder.lane_id`, `readfirstlane`, `pin_sgpr`, `to_sgpr_u32`, `wave_all`, `wave_any`, `wave_ballot`, `ds_bpermute`, `warp_shuffle_xor`, `ds_read_tr16_b64`), `core/lower_llvm.py` (intrinsic emission), `helpers/attention.py` (`warp_xor_reduce_*`).

## Wave Geometry

AMDGPU CDNA waves are 64 lanes. CK Tile / `ck_dsl` MFMA lane mappings assume wave64. There is no wave32 lowering today.

```text
lane = lane_id()       # 0..63
warp = tid / 64        # if block_size > 64
```

Workgroup size is bounded by `kernel.attrs["max_workgroup_size"]`, which lowers to the AMDGPU `"amdgpu-flat-work-group-size"="64,N"` attribute. Launching with more threads than `N` triggers `hipErrorLaunchFailure`.

## `lane_id`

```text
b.lane_id()  ->  i32 (0..63)
```

LLVM emission:

```llvm
%lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
%v  = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
```

This is a true lane index, independent of workgroup size. Prefer it over `tid & 63` when you want the AMDGPU wave lane.

## `readfirstlane`, `pin_sgpr`, `to_sgpr_u32`

The canonical "lift a wave-uniform value into scalar registers" idiom:

```python
lds_base = b.to_sgpr_u32(b.cast_i32(addr))
# subsequent uses of lds_base land in SGPR-only paths
```

`readfirstlane` returns lane 0's value broadcast across the wave. `pin_sgpr` emits `asm volatile("" : "+s"(x))`, an SGPR-class constraint that the register allocator must respect. `to_sgpr_u32` composes the two.

When to use:

- wave-uniform LDS base offsets across an unrolled K-loop;
- per-wave byte offsets in `AsyncTileLoader.bind`;
- global byte offsets that are uniform within a wave (tile origins, batch offsets).

Without `pin_sgpr`, the allocator may copy the value back into VGPRs at every use, paying a `v_readfirstlane_b32` plus an extra VGPR's worth of live-range each time.

## `wave_all`, `wave_any`, `wave_ballot`

```text
wave_all(pred)    # i32, 1 iff every lane's pred != 0
wave_any(pred)    # i32, 1 iff any lane's pred != 0
wave_ballot(pred) # i64 bitmask
```

LLVM emission uses `llvm.amdgcn.ballot.i64`. `wave_all` / `wave_any` compare the ballot mask to specific values (`-1` for all, `0` for none).

Typical use: adaptive online softmax. When `wave_all(max_diff < THRESHOLD)`, the workgroup can skip the rescale path entirely.

## `ds_bpermute`

```text
ds_bpermute(addr, data)  # both must be i32
```

Wave64 cross-lane broadcast permute using LDS as the shuffle vehicle.

```text
For each lane l:
  src_lane = (addr[l] >> 2) & 63
  result[l] = data[src_lane]
```

Notes:

- `addr` bits `[7:2]` index the source lane; high bits are ignored.
- The address bus is byte-granular, so the `>> 2` is part of the encoding, not the lowering.
- Non-i32 payloads should bitcast to i32 (or to `<2 x i32>` for 64-bit values) first.

## `warp_shuffle_xor`

```text
b.warp_shuffle_xor(v, lane_xor)
```

Convenience wrapper: lane `l` gets `v` from lane `l ^ lane_xor`. Works for f32 and i32 directly; bitcast first for half / bf16. Internally:

```text
addr = (lane_id() ^ lane_xor) << 2
result = ds_bpermute(addr, v as i32)
```

The standard butterfly reduction uses successive XORs:

```text
m = v
for off in (1, 2, 4, 8, 16, 32):
    m = max(m, warp_shuffle_xor(m, off))
```

`helpers/attention.py::warp_xor_reduce_max(b, v)` and `warp_xor_reduce_sum(b, v)` implement this for f32. Use them instead of hand-rolling unless your kernel needs a non-standard reduce.

## Register-transpose-on-store (gfx942 / CDNA3, no `ds_read_tr_b16`)

`ds_read_tr16_b64` (below) and the i8/fp8 `ds_read_tr_b8` are **gfx950-only**. On
gfx942 (CDNA3 / MI300X) there is no transpose-read instruction at all, so a
matmul that needs a transposed LDS operand (e.g. the V operand of the PV matmul
in attention) cannot get it on the read side. The working pattern — the one CK
and flash use — moves the transpose to the **store** side, in registers:

```text
buffer_load_vN (V in NATURAL [token, dim] layout, vectorized, pipelined HW DMA)
  -> in-register 2x2 transpose with v_perm  (vec_extract / vec_pack;
       __builtin_amdgcn_perm masks 0x01000504 and 0x03020706)
  -> single smem_store of the now-A-consumable layout
  -> plain conflict-free ds_read_b64 in the consumer
```

The transpose is `v_perm` (VALU), so it does **not** touch the LDS port. This is
the only conflict-free transposed-operand path on gfx942.

**Why `v_perm`, not `ds_swizzle` / `warp_shuffle_xor`, for in-register reshapes.**
`ds_swizzle` and `warp_shuffle_xor` (and `ds_bpermute`, below) are **LDS-port
ops**: they cost `lgkmcnt` and serialize against every other LDS read/write in
flight. On the gfx942 attention path, a register-V transpose built from
`ds_swizzle` measured the `lgkmcnt` wait climbing **18 → 234** and the kernel went
flat — the swizzle serialization cost exactly what the conflict removal saved.
CK's `shuffle_tile@store` dispatches the reshape **in-thread** (arranging the tile
distribution so each lane already owns its head-dim's tokens) for precisely this
reason. Use VALU `v_perm` / `permlane32_swap` for in-register reshapes; reach for
`ds_swizzle` / `ds_bpermute` only for genuine cross-lane traffic with no VALU
equivalent. See `architecture/attention_2d_gfx942_experiment_summary.md`
(Batch 3 L2, Batch 5; commits `73753189ad1`, `9cff43bdd2d`).

## The V-feed vehicle taxonomy (conflict-free transposed V on gfx942)

When a kernel needs a transposed V operand that is also conflict-free, and the
arch has no transposing async DMA and no `ds_read_tr_b16` (gfx942), there are
three vehicles. Two are traps; record them so nobody re-walks them:

| Vehicle | Correct? | Verdict | Why |
|---|---|---|---|
| (a) sync per-element `global_load` gather (transpose during the HBM read) | yes | ~5x slower | the synchronous VMEM gather stalls on HBM; not pipelined (measured ~30 TF vs L4's 148-178 on D128 fp16) |
| (b) async-DMA into an LDS staging slab, then transpose | yes | LDS-budget wall | extra LDS hop on a 64 KB/CU arch already at the occupancy threshold |
| (c) **register `buffer_load` + in-register `v_perm` transpose + single `smem_store`** | yes | **the win** | the CK / flash way: pipelined HW DMA in natural layout, VALU transpose (no LDS port), plain `ds_read_b64` consumer |

Vehicle (c) is the register-transpose-on-store pattern above. Evidence and the
units-bug that long masked (a)'s correctness: same summary doc, Batch 5.

## `ds_read_tr16_b64`

```text
b.ds_read_tr16_b64(smem, *indices, dtype=F16) -> <4 x dtype>
```

Wave64 transpose-read of a 16x16 fp16 tile from LDS. Semantics (gfx950 wave64):

```text
LDS region at smem[indices..., 0] is interpreted as 16 rows x 16 cols of fp16,
row-major, 32 bytes per row, 256 bytes total.

After the read, lane l = 16 * k_chunk + n (k_chunk in 0..3, n in 0..15) holds:
  tile[k_chunk*4 + 0, n]
  tile[k_chunk*4 + 1, n]
  tile[k_chunk*4 + 2, n]
  tile[k_chunk*4 + 3, n]
```

This is exactly the `v_mfma_f32_16x16x16_f16` per-lane B-operand. The use case is PV matmul in attention where V is in LDS row-major and we want `B[k_chunk*4 + 0..3, n]` per lane without four strided `ds_read_u16` reads. The instruction is gfx950-specific.

## `bitcast` And `vec_bitcast`

```text
bitcast(v, target_type)        # scalar bitcast (f32 <-> i32, half <-> i16, etc.)
vec_bitcast(v, target_vec)     # vector bitcast of equal size
```

`vec_bitcast` is the workhorse for the buffer-load -> MFMA bridge: the buffer-load family returns `<N x i32>`, but MFMA wants `<2N x half>`. The LLVM lowering emits `bitcast` (no `addrspacecast` involved); the AMDGPU backend folds it away.

## Patterns

### Online softmax XOR butterfly

```python
def warp_xor_reduce_max(b, v):
    m = v
    for off in (32, 16, 8, 4, 2, 1):
        other = b.warp_shuffle_xor(m, off)
        m = b.fmax(m, other)
    return m
```

This is the cheapest cross-lane max on AMDGPU — one `ds_bpermute` per step, no LDS round-trip. The XOR pattern matches the hardware's butterfly network.

### Wave-uniform LDS base hoist

```python
lds_base = b.to_sgpr_u32(b.smem_addr_of(smem))    # i64 lifted via i32
for k0 in b.unroll(0, K, block_k):
    chunk_off = b.const_i32(k0 * cols_per_chunk)
    addr = b.smem_ptr_add(lds_base, b.zext(chunk_off, I64))
    # use addr without paying readfirstlane every iter
```

### Adaptive softmax rescale skip

```python
need_rescale = b.cmp_gt(max_diff, b.const_f32(THRESHOLD))
all_safe = b.wave_all(b.lnot(need_rescale))
with b.scf_if(b.cmp_eq(all_safe, b.const_i32(0))):
    # rescale path
```

### Lane-to-output broadcast

When a single value needs to land in a specific lane (e.g. the K base for the next K tile), `ds_bpermute` with a constant `addr` shuffles it.

## Common Failure Modes

- Forgetting that `ds_bpermute` returns 0 for lanes whose source-lane bit `[6]` differs (this is the wave64 boundary). Use `warp_shuffle_xor` with a power-of-two `lane_xor` to stay within the wave.
- Bitcasting a half through an i32 without packing two halves: `ds_bpermute` only carries 32 bits per call; pack `<2 x half>` -> i32 first if you want to shuffle two halves.
- Using `wave_any` to test a boolean that has been negated incorrectly. `wave_any(pred)` is 1 iff any `pred != 0`, not iff "any pred is truthy in Python sense".
- Reading the result of `to_sgpr_u32(v)` in a path where `v` is not wave-uniform. The result will be lane 0's value broadcast everywhere; expect garbage on lanes where `v` was supposed to differ.
- Calling `ds_read_tr16_b64` on a non-16x16 tile or on data that is not row-major in the documented layout. The result is silent garbage.

## When Not To Use These

These primitives reduce instruction count and live-range pressure, but they constrain code in ways that matter for correctness:

- they assume wave64;
- the cross-lane traffic is a real cost (each `ds_bpermute` is an LDS read with full permute);
- they do not commute with arbitrary control flow (a branch can change the effective EXEC mask);
- they cannot replace barriers; they synchronize values, not memory operations.

For per-block reductions, prefer `block_lds_reduce`. For ABI-level broadcasts, prefer kernel parameters. Use these only where the analysis says they matter.
