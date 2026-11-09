# Single target continuity — 2026-10-30

Added bounded temporal association for one visible body without requiring two other characters. Requires adjacent observations within 80 ms, unchanged class, width/height within 10%, displacement no more than min(100 px at 1280 width, body height). Disabled with multiple detections or observation gaps. This is a geometry heuristic, not verified identity or measured camera motion. A similar replacement body inside the gate remains ambiguous. Large camera jumps still pause tracking.

Fallback resets the head filter and inhibits firing for that observation. Existing identity and freshness tests remain enabled. Backups: work/backup-single-target. Validation: 134 tests passed. No live-game validation performed, no driver or game settings changed.

Previous-session diagnostic: 3597 control records; 1394 target_temporarily_lost; 101 visual confirmations; median confirmation-to-movement-decision delay approximately zero, maximum 0.829 seconds. This does not measure actual kill-to-input latency. No demonstrated universal cooldown. Stutter cause remains unresolved without synchronized game-frame/input measurements.
