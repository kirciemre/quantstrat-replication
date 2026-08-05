#!/usr/bin/env bash
# Capstone: train + eval all three variants x three scenarios x N seeds,
# collect every eval into one CSV, then print the final comparison grid.
#
#   ./capstone.sh            # 5 seeds each (default)
#   ./capstone.sh 3          # 3 seeds each (quicker)
#   SEEDS=8 ./capstone.sh
#
# - hid  : one-step (GRU trains inside RL). train_hid -> eval_hid.
# - prob : two-step. train_classifier (offline, --theta-only) -> train_prob -> eval_prob.
# - reg  : two-step. train_regressor (offline) -> train_reg -> eval_reg.
# All PPO, eval seed fixed at 999 (same test set every run). Results -> capstone.csv

set -e
NSEEDS="${1:-${SEEDS:-5}}"
EVAL_SEED=999
HID_CSV=hid.csv; PROB_CSV=prob.csv; REG_CSV=reg.csv
rm -f "$HID_CSV" "$PROB_CSV" "$REG_CSV"

echo "capstone | $NSEEDS seeds/cell | eval_seed $EVAL_SEED -> {hid,prob,reg}.csv"

for sc in 1 2 3; do
  # prob & reg: offline step-1 nets are deterministic-ish; train one per seed for
  # a fair across-seed distribution (the RL seed is what drives policy variance).
  for k in $(seq 1 "$NSEEDS"); do

    echo ""; echo "############ scenario $sc | seed $k ############"

    # --- hid (one-step) ---
    python3 -m src.train.train_hid --scenario $sc --model ppo --seed $k
    python3 -m src.eval.eval_hid  --scenario $sc --model ppo --seed $EVAL_SEED \
            --results "$HID_CSV" --no-plot

    # --- prob (two-step, theta-only per the paper) ---
    python3 -m src.train.train_classifier --scenario $sc --theta-only --seed $k
    python3 -m src.train.train_prob       --scenario $sc --model ppo --theta-only --seed $k
    python3 -m src.eval.eval_prob         --scenario $sc --model ppo --theta-only \
            --seed $EVAL_SEED --results "$PROB_CSV" --no-plot

    # --- reg (two-step) ---
    python3 -m src.train.train_regressor --scenario $sc --seed $k
    python3 -m src.train.train_reg       --scenario $sc --model ppo --seed $k
    python3 -m src.eval.eval_reg         --scenario $sc --model ppo --seed $EVAL_SEED \
            --results "$REG_CSV" --no-plot

  done
done

echo ""; echo "================ FINAL GRID ================"
python3 - hid.csv prob.csv reg.csv << 'PY'
import csv, sys, numpy as np
from collections import defaultdict
cell = defaultdict(list)
for path, variant in [("hid.csv","hid"), ("prob.csv","prob"), ("reg.csv","reg")]:
    try:
        for r in csv.DictReader(open(path)):
            cell[(variant, int(r["scenario"]))].append(float(r["mean"]))
    except FileNotFoundError:
        pass

print(f'{"variant":>6} | {"Scenario 1":>16} | {"Scenario 2":>16} | {"Scenario 3":>16}')
print("-"*64)
for v in ("prob","hid","reg"):
    line = f"{v:>6} |"
    for sc in (1,2,3):
        x = np.array(cell.get((v,sc), []))
        if len(x):
            sd = x.std(ddof=1) if len(x)>1 else 0.0
            line += f" {x.mean():>6.2f}({sd:>4.2f},{len(x)}) |"
        else:
            line += f" {chr(45)*2:>16} |"
    print(line)

print("\nordering per scenario:")
for sc in (1,2,3):
    ms = {v: np.mean(cell[(v,sc)]) for v in ("prob","hid","reg") if cell.get((v,sc))}
    if ms:
        order = sorted(ms, key=ms.get, reverse=True)
        print(f"  Sc{sc}: " + " > ".join(f"{v}({ms[v]:.1f})" for v in order))

print("\npaper claims: prob > hid > reg (all scenarios)")
print("paper Sc values  prob:25.65/18.03/11.24  hid:15.70/8.08/1.29")
PY