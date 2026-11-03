#!/bin/bash
# E1 runner for the applications that have a command line: Orange (its
# bundled Python) and RapidMiner AI Studio (rapidminer-batch). R is run by
# hand with scripts/reference_r.R. Run from the E1 folder:
#   bash scripts/run_all.sh
# Each application runs twice (run 1 and run 2) per PLAN.txt controls.
# Nothing here deletes or overwrites a file that already exists.
set -u
cd "$(dirname "$0")/.." || exit 1
log() { printf '%s  %s\n' "$(date -u +%Y-%m-%dT%H:%MZ)" "$*" | tee -a NOTEBOOK.txt; }

# ---- Orange
ORANGE_APP="/Applications/Orange.app"
OPY=""
for cand in "$ORANGE_APP/Contents/MacOS/python" "$ORANGE_APP/Contents/MacOS/PythonApp" \
            "$ORANGE_APP"/Contents/Frameworks/Python.framework/Versions/*/bin/python3; do
  [ -x "$cand" ] && OPY="$cand" && break
done
if [ -z "$OPY" ]; then
  OPY=$(find "$ORANGE_APP" -type f -name 'python3*' -perm -u+x 2>/dev/null | head -1)
fi
if [ -n "$OPY" ] && "$OPY" -c 'import Orange' 2>/dev/null; then
  log "Orange python: $OPY ($("$OPY" -c 'import Orange;print(Orange.version.version)'))"
  for run in 1 2; do
    if ls reference_outputs/orange/R1_orange_run${run}_predictions.csv >/dev/null 2>&1; then
      log "Orange run $run already present, skipped"; continue
    fi
    "$OPY" scripts/reference_orange.py "$run" && log "Orange run $run done" || log "Orange run $run FAILED"
  done
else
  log "Orange python not found under $ORANGE_APP; try: conda run -n <env with orange3> python scripts/reference_orange.py 1"
fi

# ---- RapidMiner AI Studio, batch mode on the saved process
RM_APP=$(ls -d /Applications/AI\ Studio*.app 2>/dev/null | head -1)
RMB=$(find "$RM_APP" -name 'rapidminer-batch.sh' 2>/dev/null | head -1)
if [ -n "$RMB" ]; then
  log "AI Studio batch script: $RMB"
  for run in 1 2; do
    if ls reference_outputs/aistudio/R1_aistudio_run${run}_predictions.csv >/dev/null 2>&1; then
      log "AI Studio run $run already present, skipped"; continue
    fi
    sed "s|__RUN__|$run|g; s|__E1__|$PWD|g" scripts/E1_aistudio_template.rmp > "reference_outputs/aistudio/E1_run${run}.rmp"
    "$RMB" -f "reference_outputs/aistudio/E1_run${run}.rmp" \
      && log "AI Studio run $run done" || log "AI Studio run $run FAILED (open reference_outputs/aistudio/E1_run${run}.rmp in AI Studio and run it there)"
  done
else
  log "rapidminer-batch.sh not found under $RM_APP; open scripts/E1_aistudio_template.rmp in AI Studio after replacing __RUN__ and __E1__"
fi
ls -la reference_outputs/orange reference_outputs/aistudio 2>/dev/null
