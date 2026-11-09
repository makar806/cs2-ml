# Research stage 1 — 2026-10-31

Label Studio uses the existing .label-studio database and media, localhost:8080. Do not export/retrain from an actively changing database without a consistent snapshot.

First separate localization from control. scripts/record_localization.py starts only the capture/inference process. It never creates Mouse, Engine or a control loop and does not install GSI. Scenario names are manual metadata, not automatic verification of movement conditions. Record static, player-motion, camera-motion and both-motion with the same scene and graphics. Use at least 90 seconds to allow warmup. Close the ordinary MVP first to avoid duplicate inference load.

Example: .venv/Scripts/python.exe scripts/record_localization.py --scenario static --duration 90 --graphics low

Frames and complete raw detection packets are saved under runs/localization. This stage records predictions; it does not establish localization accuracy without reference annotations. Moving-camera coordinate variation alone is not jitter.

scripts/report_latency.py SESSION --out REPORT.json reports p50/p95/p99, sample counts and maximum. Source intervals include pauses/focus gaps; they must not be presented as game frametimes. Cached kill-feed timing samples may be repeated. Reported packet age differs from capture-to-action age.

scripts/mvp_one_euro.py implements an independent 1-Euro filter. It is not layered into live control. Tests establish numerical behavior, not improvement on CS2. Parameters need selection on training sessions and evaluation on held-out recordings. Camera motion must be compensated before interpreting pixel trajectories.

Next: replay/comparison of raw points, current HeadFilter and 1-Euro with reference trajectories; then offline SAMEO. SAMEO GUI toggle is planned only with a functional backend, load/unload lifecycle and measured resource cost. No claim that SAMEO is integrated or weights have been downloaded. Weapon HUD matching remains an unfinished separate experiment and is not enabled in runtime.


Update: v3.17 adds optional live 1-Euro selection for the recorded controller; see docs/v3.17.md. Earlier offline-only status above describes the initial research stage.
