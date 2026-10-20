# Prosody metric sources

Frozen from the user-supplied benchmark and its feature operator on 2026-10-20.
No runtime dependency on these shared filesystem paths.

- `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/cgy/codex/rl_data_recall_20261018/bench_triplet/metrics/prosody_diff.py`
  SHA-256: `aec119c5e34068731f02a960e37aec767687a2feec3ed13e9c39b7e84af9f9ae`
- `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/btjiang/MOSS-TTS-Data/scripts/stage_Prosody/prosody_operator.py`
  SHA-256: `bdb2ee517dc53a3cec1329d91aa50af4e5f4b1454c93d4d2c17d57a3561a3c03`
- `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/cgy/all-data-attrs/prosody-recall/_global/prosody_global_four_profile_tertile_v2.json`
  SHA-256: `e66996631f6a85603019052c8907fb208905fc91dc41ada7ef7709d487db54aa`

`_prosody_features.py` includes only `decode_audio`, `_runs`, `_finite_stats`,
`_tail_track_delta` and `waveform_features`, with formatting changes only.
`prosody_stats.json` preserves the zh/en medians and IQR/1.349 scales exactly;
cohort details, quantiles and bucketing metadata are omitted.
The YIN window is 2048 samples; the energy/segmentation window is 400 samples,
both at a hop of 160 samples and a sample rate of 16 kHz.
