# Automatic OPLS result analysis

The analyzer reads the results ZIP from `run_opls_benchmark.sh`. It runs locally,
requires only NumPy, and does not need another Unity job. It never runs Python
files contained in the ZIP or recomputes force-field energies.

After downloading the results into this project, run in PowerShell:

```powershell
.\.runtime\python\python.exe scripts/analyze_opls_benchmark.py benchmark_results.zip --output outputs/opls_benchmark_analysis_1820024
```

Use the actual downloaded filename if Windows added `(1)`; quote filenames with
spaces. The output directory must be new to preserve earlier reviews.

Outputs:

- `ANALYSIS.md`: short comparison, completion counts, timing, and audit problems.
- `analysis.json`: per-input and per-pair differences, failures, and detailed checks.

The analyzer checks the frozen selection and candidate plan, scheduled seeds,
parameter hashes, atom ordering, saved raw/relaxed coordinates, relaxation flags,
guidance noise gates and displacement caps, and report totals. If these checks
fail, its Markdown report withholds performance interpretation.

The comparison shows raw energy/force differences, accepted relaxed energy
differences, repair displacement, and generation-time ratios. Setup timing is
separate. A partial run retains the original planned denominator. Assessable
pairs are explicitly counted; a failed relaxation does not erase its raw result.

Training inputs, several seeds, and repeated candidate structures are not
independent validation molecules. These are descriptive results, not accuracy
estimates or significance tests. Coordinate-inferred graph agreement ignores
stereochemistry. The audit checks internal consistency of saved artifacts,
not provenance authenticity or experimental physical correctness.

No early-rejection threshold, model weights, or force-field settings are changed.
