"""
run_pipeline.py -- One command to reproduce the whole project end to end.

    python run_pipeline.py

Runs every stage in dependency order with the fixed seed, so a reviewer can
regenerate all data, models, metrics, figures, and explanations from scratch.
The Streamlit dashboard is launched separately:  streamlit run dashboard/app.py
"""

import subprocess
import sys
import time

STAGES = [
    ("Phase 1  synthetic data generator", "generator/generator.py"),
    ("Phase 2  incremental feature layer", "models/features.py"),
    ("Phase 2  Isolation-Forest baseline", "models/baseline.py"),
    ("Phase 3  GRU sequence autoencoder + fusion", "models/sequence.py"),
    ("Phase 3  attack-type classifier + novel class", "models/classifier.py"),
    ("Phase 4  per-alert explanations", "models/explain.py"),
    ("Phase 4  full metrics suite + figures", "models/metrics.py"),
]


def main():
    t0 = time.time()
    for i, (name, script) in enumerate(STAGES, 1):
        print(f"\n{'#'*72}\n# [{i}/{len(STAGES)}] {name}\n{'#'*72}")
        r = subprocess.run([sys.executable, script])
        if r.returncode != 0:
            print(f"\n!! stage failed: {script} (exit {r.returncode}) -- stopping.")
            sys.exit(r.returncode)
    print(f"\n{'='*72}\nPIPELINE COMPLETE in {time.time()-t0:.0f}s.")
    print("Launch the analyst dashboard with:  streamlit run dashboard/app.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
