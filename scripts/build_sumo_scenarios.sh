#!/usr/bin/env bash
# Build the shared four-way SUMO network from the committed XML *sources*
# and verify the generated traffic-light program satisfies the config
# contract. Requires SUMO's netconvert on PATH (a SUMO install).
#
#   Run from the repo root:   bash scripts/build_sumo_scenarios.sh
#
# Committed sources (simulation/scenarios/):
#   intersection.nod.xml  node definitions (junction J = traffic_light "tl_1")
#   intersection.edg.xml  edges (N0/S0/E0/W0 approaches + N1..W1 outgoing)
#   intersection.add.xml  5-phase tlLogic matching configs/signal.yaml
#
# Produces (gitignored): simulation/scenarios/intersection.net.xml
set -euo pipefail

SCEN_DIR="simulation/scenarios"
NET_FILE="$SCEN_DIR/intersection.net.xml"

echo "=== netconvert: $SCEN_DIR -> $NET_FILE ==="
netconvert \
  --node-files "$SCEN_DIR/intersection.nod.xml" \
  --edge-files "$SCEN_DIR/intersection.edg.xml" \
  --additional-files "$SCEN_DIR/intersection.add.xml" \
  --output-file "$NET_FILE"

echo "=== verifying tl_1 program vs configs/signal.yaml ==="
python3 - "$NET_FILE" <<'PY'
import sys
import xml.etree.ElementTree as ET
import yaml

net_path = sys.argv[1]
root = ET.parse(net_path).getroot()
tls = [n for n in root.iter() if n.tag == "tlLogic" and n.get("id") == "tl_1"]
if not tls:
    sys.exit("error: generated network has no tlLogic id='tl_1'")
n_phases = len(list(tls[0]))
print(f"ok: tl_1 has {n_phases} phases")

cfg = yaml.safe_load(open("configs/signal.yaml")) or {}
sim = cfg.get("simulation") or {}
idx = []
idx.append(sim.get("phase_index_all_red", 0))
for k in ("phase_index_green", "phase_index_yellow"):
    idx.extend((sim.get(k) or {}).values())
max_used = max(idx)
if n_phases <= max_used:
    sys.exit(
        f"error: config references phase index {max_used} but tl_1 has only "
        f"{n_phases} phases; align simulation/scenarios/intersection.add.xml "
        f"with the netconvert-generated program"
    )
print(f"ok: tl_1 (indices 0..{n_phases-1}) covers all config phase indices")
PY

echo "done. Run experiments next with:"
echo "  python scripts/run_experiments.py --help"