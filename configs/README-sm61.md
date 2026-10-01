# Strata on GTX 1080 Ti (sm_61) — variant configs

Configs for running the Qwen3.8-Flash-Next 125.7B MoE Q2_0 pack
(`/mnt/models2/strata-q2_0-pack/`) on a dual-GPU Pascal box
(2x GTX 1080 Ti 11 GiB). Base release **0.1.30** + two patches on top:

- `STRATA_EXPERIMENTAL_SM60` is propagated as a compile definition and the
  `cc < 75` device guard is skipped when it is set (commit 33d9a9f).
- `tools/iq_pack.py` reads GGUF shards with an `os.pread` retry loop instead of
  `np.memmap` (SIGBUS on flaky SATA links kills the process otherwise), and
  supports resuming a partially written `experts.bin` (commit cc12d0a).

Build: `cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES=61 -DSTRATA_EXPERIMENTAL_SM60=ON ...`

## Variants (measured 2026-10-01)

Launch identical for all three — point `serve/server.py` at the config:

```
python3 serve/server.py --engine strata --config configs/sm61-single-gpu.json \
    --host 0.0.0.0 --port 8095
```

| Config | What the 2nd GPU does | Decode cold-cache | Decode warm-cache | Prefill warm |
|---|---|---|---|---|
| `sm61-single-gpu.json` | idle | 0.91 tok/s | 6.90 tok/s | 17.2 tok/s |
| `sm61-2gpu-device1.json` | 3600 extra expert slots (device1) | **1.35 tok/s (+48 %)** | **7.09-7.33 tok/s** | **17.2 tok/s** |
| `sm61-2gpu-expert-cache.json` | same tier + `remote-placement layer` | — | was measured slower | — |
| `sm61-2gpu-layer-split.json` | pipeline: layers 19-47 + head on CUDA1 | — | was measured slower | — |

**Realistic-prompt A/B (2026-10-01, identical decode streams, cold = `drop_caches`)** — the
earlier "single GPU fastest" table was confounded by a degenerate 64x-"9" prompt whose token
stream differed per config, so its tok/s numbers across configs were not comparable. The
cold-cache result is the one production traffic sees (page cache of the 34 GB experts.bin
gets evicted; 31 GB RAM). `--expert-cache-per-layer` and `--no-prefill-borrow` measured
slower and are not exposed here; the layer_next_ admission bug they uncovered is fixed in
commit 7d0f9f9. The measured numbers make the
trade-off explicit:

- `--layer-split auto --split-device CUDA0,CUDA1` — a capacity option (KV/expert
  cache spread over both GPUs, larger `--max-context`), never a speedup; the
  pipeline hand-off costs ~12 ms/window. Also: helper caches
  (`--expert-cache-remote`) cannot combine with layer splits.
- `--expert-cache-device1 N` — CUDA0 keeps dense/state/MTP + its 4400-slot
  auto cache; CUDA1 gets the *rest of the profile* (8000 ranked pairs total,
  no eviction, so N above ~3600 is clamped). The full-profile hit rate rises
  to 72-80%, but remote hits return through pinned host rows
  (`--expert-cache-remote-placement layer` halves the loss vs the default
  `stripe` and is what the config pins).

## Operational notes

- Graceful stop: `SIGTERM` works (engine installs a handler); `SIGINT` is
  *ignored* when the server is launched via `nohup` (ignored dispositions are
  inherited through `execve`).
- First start fills the CUDA0 expert cache from the profile (~1 min, machine
  briefly slow while experts.bin is paged in).
- `--max-context 8192` and `--kv int8` are tuned for this 11 GiB card; scale
  with care — the 720 MiB+73 MiB reserve is already tight.