#!/bin/bash
# Run run_sweep.sh back to back until stopped, always with the hybrid rain
# model, raising the seed each round:
#
#   hybrid seed 1, hybrid seed 2, hybrid seed 3, ...
#
# The model is fixed here on purpose; to sweep another model, run
# `RAIN_MODEL=<model> SEED=<n> ros2 run thesis_rain_sim run_sweep.sh` by hand.
#
#   ros2 run thesis_rain_sim overnight.sh [WORLD]       # WORLD default rain_yard
#
# Environment:
# WEATHER, RAIN_LEVELS, FOG_LEVELS, DUST_LEVELS
#                 which weathers each sweep runs, passed to run_sweep.sh (see
#                 there). Default (none of them set): one pass on each condition
#                 alone, then one pass with all three active, light and heavy:
#                   calm light heavy fog_light fog_heavy dust_light dust_heavy
#                   rain_light+fog_light+dust_light rain_heavy+fog_heavy+dust_heavy
#                 (9 weathers x raw / denoised = 18 runs per sweep, ~4 h).
# FIRST_SEED      seed of round 1 (default 1)
# MISSION_LOOPS   passed to run_sweep.sh (default 2)
# MAX_BAD         stop after this many sweeps in a row with no successful goal (default 3)
#
# Each sweep writes the usual ~/rain_sim_results/<world>_YYYYMMDD_HHMM/ folder.
# The night's index lives in ~/rain_sim_results/overnight_YYYYMMDD_HHMM/:
#   index.csv      one row per sweep: folder, model, seed, times, exit code,
#                  succeeded/attempted goal attempts per condition/pipeline
#   overnight.log  timestamped start / finish / failure lines, and one line per
#                  run as it starts (also printed to the terminal):
#                  [..] round 1 seed 1: run 5/18  fog_light / raw
#
# Stopping:
#   touch ~/rain_sim_results/overnight_*/STOP   finish the current sweep, then exit
#   Ctrl-C, or kill -TERM <pid>                 tear the current sweep down now
#                                              (that sweep's folder is incomplete)
set -u
WORLD="${1:-rain_yard}"
MODEL=hybrid
FIRST_SEED="${FIRST_SEED:-1}"
LOOPS="${MISSION_LOOPS:-2}"
MAX_BAD="${MAX_BAD:-3}"
if [ -z "${WEATHER:-}${RAIN_LEVELS:-}${FOG_LEVELS:-}${DUST_LEVELS:-}" ]; then
  export WEATHER="calm light heavy fog_light fog_heavy dust_light dust_heavy \
rain_light+fog_light+dust_light rain_heavy+fog_heavy+dust_heavy"
fi
if [ -n "${WEATHER:-}" ]; then
  WEATHER_DESC="$WEATHER"
else
  WEATHER_DESC="rain=[${RAIN_LEVELS:-calm light heavy}] fog=[${FOG_LEVELS:-off}] dust=[${DUST_LEVELS:-off}]"
fi
RESULTS="$HOME/rain_sim_results"
NIGHT="$RESULTS/overnight_$(date +%Y%m%d_%H%M)"
INDEX="$NIGHT/index.csv"
LOG="$NIGHT/overnight.log"

# Call the installed script directly (not via `ros2 run`), so $! is the sweep's
# own bash and SIGTERM reaches its teardown trap.
PREFIX="$(ros2 pkg prefix thesis_rain_sim 2>/dev/null)" || {
  echo "thesis_rain_sim not found - source ~/ros2_ws/install/setup.bash first" >&2; exit 1; }
SWEEP_CMD="${SWEEP_CMD:-$PREFIX/lib/thesis_rain_sim/run_sweep.sh}"

mkdir -p "$NIGHT"
log () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if pgrep -f "gz sim" >/dev/null; then
  echo "A gz sim is already running (pgrep -af 'gz sim'). Run one simulation at a time." >&2
  exit 1
fi

echo "sweep_dir,model,seed,round,start,end,duration_min,exit_code,status,goal_attempts_succeeded_per_run,weather" > "$INDEX"

SP=""
STOPPING=0
on_signal () {
  STOPPING=1
  log "signal received - stopping the current sweep and exiting"
  [ -n "$SP" ] && kill -TERM "$SP" 2>/dev/null
}
trap on_signal INT TERM

# "calm/raw 8/9; calm/denoised 7/8; ..." from a missions.csv
# (columns: loop,goal_idx,condition,pipeline,...,result,...; one row per attempt)
goal_summary () {
  [ -f "$1" ] || { echo "no missions.csv"; return; }
  awk -F, 'NR > 1 { k = $3 "/" $4; n[k]++; if ($7 == "SUCCEEDED") s[k]++
                    if (!(k in seen)) { seen[k] = 1; order[++m] = k } }
           END { for (i = 1; i <= m; i++) { k = order[i]
                   printf "%s%s %d/%d", (i > 1 ? "; " : ""), k, s[k] + 0, n[k] } }' "$1"
}

log "overnight start: world=$WORLD model=$MODEL first_seed=$FIRST_SEED loops=$LOOPS weather: $WEATHER_DESC"
log "index: $INDEX   stop after current sweep: touch $NIGHT/STOP"

ROUND=0
NSWEEPS=0
BAD=0
while [ $STOPPING -eq 0 ]; do
  ROUND=$(( ROUND + 1 ))
  SEED=$(( FIRST_SEED + ROUND - 1 ))
  if [ -e "$NIGHT/STOP" ]; then log "STOP file found - exiting"; break; fi

  # run_sweep.sh's folder name has minute resolution; never reuse one
  OUTDIR="$RESULTS/${WORLD}_$(date +%Y%m%d_%H%M)"
  while [ -e "$OUTDIR" ]; do sleep 5; OUTDIR="$RESULTS/${WORLD}_$(date +%Y%m%d_%H%M)"; done

  START="$(date -Iseconds)"; T0=$SECONDS
  log "round $ROUND: start model=$MODEL seed=$SEED -> $OUTDIR"
  mkdir -p "$OUTDIR"
  RAIN_MODEL="$MODEL" SEED="$SEED" "$SWEEP_CMD" "$WORLD" "$OUTDIR" "$LOOPS" \
    > "$OUTDIR/sweep_stdout.log" 2>&1 &
  SP=$!
  # one line per run as the sweep starts it, from its " RUN i/N ..." header;
  # tail --pid exits when the sweep does
  tail --pid="$SP" -n +1 -F "$OUTDIR/sweep_stdout.log" 2>/dev/null \
    | sed -un 's/^ RUN \([0-9]*\/[0-9]*\) .* weather=\([^ ]*\) .* pipeline=\([^ ]*\) .*/\1 \2 \3/p' \
    | while read -r IDX W P; do log "round $ROUND seed $SEED: run $IDX  $W / $P"; done &
  TP=$!
  wait "$SP"; RC=$?
  # a trapped signal interrupts `wait`; wait again for the teardown to finish
  while kill -0 "$SP" 2>/dev/null; do wait "$SP"; RC=$?; done
  SP=""
  wait "$TP" 2>/dev/null
  NSWEEPS=$(( NSWEEPS + 1 ))

  MIN=$(( (SECONDS - T0) / 60 ))
  GOALS="$(goal_summary "$OUTDIR/missions.csv")"
  if [ $STOPPING -eq 1 ]; then STATUS=interrupted
  elif grep -q SUCCEEDED "$OUTDIR/missions.csv" 2>/dev/null; then STATUS=ok; BAD=0
  else STATUS=no_successful_goal; BAD=$(( BAD + 1 )); fi
  echo "$OUTDIR,$MODEL,$SEED,$ROUND,$START,$(date -Iseconds),$MIN,$RC,$STATUS,\"$GOALS\",\"$WEATHER_DESC\"" >> "$INDEX"
  log "round $ROUND: done  model=$MODEL seed=$SEED ${MIN} min rc=$RC $STATUS | $GOALS"

  if pgrep -f "gz sim" >/dev/null; then
    log "WARNING: gz sim still alive after the sweep - killing it"
    pkill -9 -f "gz sim"; sleep 5
  fi
  if [ $BAD -ge "$MAX_BAD" ]; then
    log "$BAD sweeps in a row without a successful goal - something is broken, stopping"
    STOPPING=1
  fi
done
log "overnight end after $NSWEEPS sweep(s). Index: $INDEX"
