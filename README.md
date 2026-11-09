# factory-anomaly-detector

Unsupervised anomaly detection for a CNC / robot manufacturing cell —
vibration, temperature, and torque streams with injected faults
(bearing wear, overheating, tool breakage). Built for Prachi Payasi's
Spring 2027 FTR project, "AI implementation for smart manufacturing"
(Prof. Martin Byung-Guk Jun / Tyson Oh): industrial robots, data
analysis and labelling, AI model development, and digitalization of
physical assets.

Two detectors are implemented from scratch in NumPy (no sklearn) and
compared on the same labelled synthetic data:

1. **Isolation Forest** — an ensemble of random isolation trees;
   anomalies are isolated in few splits, so a short average path
   length means "anomalous".
2. **Autoencoder** — a small MLP (`6 → 8 → 2 → 8 → 6`) trained with
   Adam on normal data only; the anomaly score is the per-sample
   mean-squared reconstruction error.

Both are fit on normal data only (semi-supervised, as in a real
factory where labelled faults are scarce) and set their decision
threshold at the 90th percentile of training scores — a ~10%
false-alarm rate on normal windows, the right trade-off when a
missed fault costs far more than a checked alarm.

## How to run

```bash
pip install -r requirements.txt   # numpy only
python3 demo.py                                  # full comparison + fault-onset stream
python3 -m unittest discover -s tests -v         # unit tests
```

## Results (measured by running `python3 demo.py`)

Labelled test set: 800 normal + 300 anomalous windows
(100 each of bearing wear, overheating, tool breakage), seed 42.

| Detector | Precision | Recall | F1 | Bearing wear | Overheating | Tool breakage |
|---|---|---|---|---|---|---|
| Isolation Forest | 0.798 | 0.853 | 0.824 | 0.88 | 0.70 | 0.98 |
| Autoencoder | 0.739 | 0.880 | 0.804 | 0.78 | 0.97 | 0.89 |

Fault-onset stream (bearing wear starts at window 300 of 500):
both detectors fire at window 300 — **0 windows of detection delay**.
Isolation Forest catches 86% of the faulty windows after onset;
autoencoder catches 73%.

Takeaway: the detectors are complementary — the autoencoder is far
better at overheating (0.97 vs 0.70 recall: the fault breaks the
normal temperature/torque correlation, which the autoencoder's
compressed bottleneck cannot reconstruct), while the Isolation
Forest is better at bearing wear and tool breakage. In production,
run both and alarm on either.

## Tests

`python3 -m unittest discover -s tests -v` — **11/11 pass**, covering
the data generator (shapes, labels, determinism, fault shifts, stream
onset), both detectors (precision/recall/F1 above 0.70, per-fault
recall above 0.50 for every fault type, reconstruction error falling
with longer training), and the evaluation metrics (hand-checked
precision/recall/F1 and a perfect-prediction case).

## Files

- `factory_anomaly/generator.py` — synthetic CNC/robot-cell sensor
  streams (6 features: vibration RMS/kurtosis/peak, temperature,
  torque mean/std) with three injected fault modes.
- `factory_anomaly/detectors.py` — Isolation Forest and NumPy
  autoencoder, from scratch.
- `factory_anomaly/evaluate.py` — precision/recall/F1, confusion
  counts, per-fault recall.
- `demo.py` — trains both detectors, compares them on labelled data,
  and runs the fault-onset stream demo.
- `tests/test_factory_anomaly.py` — unit tests (11 tests).

## Application talking point

"Built an unsupervised anomaly detector for a CNC/robot cell that
catches bearing wear, overheating, and tool breakage from vibration,
temperature, and torque streams — Isolation Forest F1 0.82 and a
NumPy autoencoder F1 0.80 on labelled synthetic data, with zero-delay
detection of fault onset in a live stream."
