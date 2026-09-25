Differential test E1, DiGiTerra side rerun of 2026-09-25 after the change to scikit-learn defaults.

What was rerun: scripts/run_digiterra.py from docs/differential_test_E1, unchanged, against the
DiGiTerra tree of this commit (0b0ae6e with the Polynomial, Perceptron and Birch changes that
follow it), in a Linux x86_64 container with Python 3.11.15, scikit-learn 1.9.0, numpy 1.26.4 and
pandas 2.3.3 (digiterra_outputs/versions.txt). The reference programs (R, Excel, Orange, AI Studio)
and the arbiter were not rerun; their saved outputs in docs/differential_test_E1 do not depend on
DiGiTerra. scripts/compare.py was then run over the new digiterra_outputs and the saved reference
and arbiter outputs.

Input checks: 5 of 5 model cases passed (rerun log line "input check passed").
Result: RESULTS.csv holds 2,436 cells; DiGiTerra has 530, classed 457 same, 56 rounded 6 dp,
10 near zero, 7 last digit and 0 other, the counts of the record in docs/differential_test_E1.
Every DiGiTerra value in RESULTS.csv equals the value in the recorded RESULTS.csv exactly
(1,064 matched case, file and term rows, maximum absolute difference 0.0). RESULT_2026-09-25.txt
reports the set as COMPLETE.

The deposited record in docs/differential_test_E1 is kept as it ran; this folder holds only the
rerun of the DiGiTerra side and the comparison over it. digiterra_outputs holds the same files as
the record's digiterra_outputs; model.skops, model_card.html and the correlation responses the
rerun also wrote are left out, as in the record.
