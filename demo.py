"""Demo / experiment: train both detectors on normal data only, then
compare precision / recall / F1 on a labelled test set with injected
faults (bearing wear, overheating, tool breakage), plus a time-ordered
stream showing fault-onset detection.

Usage:  python3 demo.py
"""

import numpy as np

from factory_anomaly.generator import (
    generate_dataset, generate_stream, standardize, FAULT_TYPES,
)
from factory_anomaly.detectors import IsolationForestDetector, AutoencoderDetector
from factory_anomaly.evaluate import precision_recall_f1, per_fault_recall


def run(seed=42):
    # Train on normal data only; test on a fresh labelled mix.
    X_train, _, _ = generate_dataset(n_normal=1200, n_anomaly=0, seed=seed)
    X_test, y_test, faults = generate_dataset(
        n_normal=800, n_anomaly=300, seed=seed + 100)
    X_train_s, X_test_s, _, _ = standardize(X_train, X_test)

    results = {}
    for name, detector in [
        ("Isolation Forest", IsolationForestDetector(n_trees=100, seed=seed)),
        ("Autoencoder", AutoencoderDetector(epochs=300, seed=seed)),
    ]:
        detector.fit(X_train_s)
        y_pred = detector.predict(X_test_s)
        metrics = precision_recall_f1(y_test, y_pred)
        metrics["per_fault_recall"] = per_fault_recall(y_test, y_pred, faults)
        results[name] = metrics

    # Stream demo: bearing wear starts at window 300.
    X_stream, y_stream, _ = generate_stream(
        n_windows=500, fault_at=300, fault="bearing_wear", seed=seed)
    _, X_stream_s, _, _ = standardize(X_train, X_stream)
    stream = {}
    for name, detector in [
        ("Isolation Forest", IsolationForestDetector(n_trees=100, seed=seed)),
        ("Autoencoder", AutoencoderDetector(epochs=300, seed=seed)),
    ]:
        detector.fit(X_train_s)
        pred = detector.predict(X_stream_s)
        # First window where the detector fires after onset (or None).
        fired = np.where(pred[300:] == 1)[0]
        stream[name] = {
            "first_detection_window": int(300 + fired[0]) if len(fired) else None,
            "detection_delay_windows": int(fired[0]) if len(fired) else None,
            "false_alarms_before_onset": int(pred[:300].sum()),
            "fault_recall_in_stream": float(pred[300:].mean()),
        }
    return results, stream


def main():
    results, stream = run()
    print("Factory anomaly detection - labelled test set")
    print("  test: 800 normal + 300 anomalous "
          "(100 each: bearing wear, overheating, tool breakage)")
    print()
    for name, m in results.items():
        print(f"{name}")
        print(f"  precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
              f"F1 {m['f1']:.3f}   (TP {m['tp']} FP {m['fp']} "
              f"FN {m['fn']} TN {m['tn']})")
        for fault in FAULT_TYPES:
            print(f"    recall on {fault:<15} {m['per_fault_recall'][fault]:.3f}")
    print()
    print("Fault-onset stream (bearing wear starts at window 300 of 500)")
    for name, s in stream.items():
        print(f"  {name}: first detection window "
              f"{s['first_detection_window']} "
              f"(delay {s['detection_delay_windows']} windows), "
              f"{s['false_alarms_before_onset']} false alarms before onset, "
              f"fault recall {s['fault_recall_in_stream']:.3f}")


if __name__ == "__main__":
    main()
