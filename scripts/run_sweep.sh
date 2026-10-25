#!/bin/bash
# Batch-run the weather / denoising navigation experiment across
# {weathers} x {raw, denoised} for one world, then analyse.
#
#   ros2 run thesis_rain_sim run_sweep.sh [WORLD] [OUTDIR] [MISSION_LOOPS]
#
# WORLD          rain_room (default) | rain_yard
# OUTDIR         where CSVs land (default: ~/rain_sim_results/<world>_<date>)
# MISSION_LOOPS  laps of the goal set per run (default 2)
#
# Environment:
# RAIN_MODEL     hybrid (default) | goodin2019 | alavi2025 | legacy
#                (fog / dust need goodin2019, alavi2025 or hybrid)
# SEED           weather_model_node RNG seed (default 1)
# Which weathers to run - either every combination of
# RAIN_LEVELS    rain levels (default "calm light heavy")
# FOG_LEVELS     fog levels  (default "off")
# DUST_LEVELS    dust levels (default "off")
# or an explicit list (takes precedence):
# WEATHER        weather labels, e.g. "calm fog_light fog_heavy dust_heavy rain_heavy+fog_light"
#                (bare calm/light/heavy = rain only; see thesis_rain_sim/weather.py)
#
#   RAIN_MODEL=hybrid SEED=2 ros2 run thesis_rain_sim run_sweep.sh rain_yard
#   RAIN_MODEL=hybrid RAIN_LEVELS=calm FOG_LEVELS="off light heavy" run_sweep.sh   # fog only
#   RAIN_MODEL=hybrid WEATHER="calm dust_light dust_heavy" run_sweep.sh rain_yard
#
# With none of the weather variables set the sweep is calm / light / heavy
# rain, exactly as before fog and dust existed.
#
# Each run: launch sim+weather+denoiser+SLAM+Nav2 headless, wait for Nav2,
# execute the goal missions, tear down. ~4-8 min per run.
# A live progress bar shows goals done in the current run and overall.
# Ctrl-C tears everything down and stops the whole sweep.
# OUTDIR/sweep_info.txt records world, model, seed, weathers, start time and
# the package's git commit.
set -u
WORLD="${1:-rain_room}"
STAMP="$(date +%Y%m%d_%H%M)"
OUTDIR="${2:-$HOME/rain_sim_results/${WORLD}_${STAMP}}"
LOOPS="${3:-2}"
RAIN_MODEL="${RAIN_MODEL:-hybrid}"
SEED="${SEED:-1}"

# --- weathers to run: labels, validated before anything starts -------------
weather_cli () { python3 -m thesis_rain_sim.weather "$@"; }
CONDS=()
if [ -n "${WEATHER:-}" ]; then
  read -r -a CONDS <<< "$WEATHER"
else
  for r in ${RAIN_LEVELS:-calm light heavy}; do
    for f in ${FOG_LEVELS:-off}; do
      for d in ${DUST_LEVELS:-off}; do
        label="$(weather_cli label "$r" "$f" "$d")" || exit 1
        [[ " ${CONDS[*]} " == *" $label "* ]] || CONDS+=("$label")
      done
    done
  done
fi
[ ${#CONDS[@]} -gt 0 ] || { echo "no weathers to run" >&2; exit 1; }
for COND in "${CONDS[@]}"; do
  parsed="$(weather_cli parse "$COND")" || exit 1
  read -r R F D <<< "$parsed"
  if [ "$RAIN_MODEL" = "legacy" ] && { [ "$F" != off ] || [ "$D" != off ]; }; then
    echo "weather '$COND' has fog/dust, which need RAIN_MODEL=goodin2019|alavi2025|hybrid" >&2
    exit 1
  fi
done
mkdir -p "$OUTDIR"

# git commit of the package source: the script's own tree (symlink-install),
# else <ws>/src/thesis_rain_sim next to the install prefix.
git_commit () {
  local d cands=()
  cands+=("$(dirname "$(readlink -f "$0")")/..")
  local prefix
  prefix="$(ros2 pkg prefix thesis_rain_sim 2>/dev/null)" \
    && cands+=("$prefix/../../src/thesis_rain_sim")
  for d in "${cands[@]}"; do
    if git -C "$d" rev-parse HEAD >/dev/null 2>&1; then
      local dirty=""
      [ -n "$(git -C "$d" status --porcelain 2>/dev/null)" ] && dirty=" (uncommitted changes)"
      echo "$(git -C "$d" rev-parse HEAD)$dirty"
      return
    fi
  done
  echo "n/a (package is not a git repository)"
}

{
  echo "world:       $WORLD"
  echo "rain_model:  $RAIN_MODEL"
  echo "seed:        $SEED"
  echo "weathers:    ${CONDS[*]}"
  echo "loops:       $LOOPS"
  echo "start_time:  $(date -Iseconds)"
  echo "git_commit:  $(git_commit)"
} > "$OUTDIR/sweep_info.txt"
echo "SWEEP  world=$WORLD  model=$RAIN_MODEL  seed=$SEED  loops=$LOOPS  weathers=[${CONDS[*]}]  -> $OUTDIR"

LOG_CSV="$OUTDIR/nav_error_log.csv"
MIS_CSV="$OUTDIR/missions.csv"
rm -f "$LOG_CSV" "$MIS_CSV"

# Goals must be reachable free space. The originals had (-6,6) in rain_yard at
# the centre of the shed (3.0 x 2.2 box at -6,6 -> spans y 4.5..7.5) and (2.5,2.5)
# in rain_room exactly on obstacle_box_1's corner (1 m box at 2,2) - both
# unreachable, so every loop burned all retries on them.
if [ "$WORLD" = "rain_yard" ]; then
  GOALS="[6.0,6.0,0.0, -6.0,9.0,3.14, -6.0,-6.0,0.0, 6.0,-6.0,1.57]"
else
  GOALS="[3.5,3.5,0.0, -2.5,2.5,3.14, -2.5,-2.5,0.0, 2.5,-2.5,1.57]"
fi
N_GOALS=4
MAX_RETRIES=3
GOAL_TIMEOUT=240   # s per goal attempt before run_missions cancels it
# Hard cap on a whole mission run, in case anything still wedges:
# every attempt timing out, plus settle/recovery slack and startup.
MISSION_WATCHDOG=$(( LOOPS * N_GOALS * (MAX_RETRIES + 1) * (GOAL_TIMEOUT + 30) + 180 ))

PIPES=(raw denoised)
TOTAL_RUNS=$(( ${#CONDS[@]} * ${#PIPES[@]} ))
RUN_IDX=0
SWEEP_T0=$SECONDS
LP=""
MP=""

if [ -t 1 ]; then TTY=1; else TTY=0; fi

fmt_time () { printf '%02d:%02d' $(( $1 / 60 )) $(( $1 % 60 )); }

# draw_bar DONE TOTAL TEXT - redraws one status line in place (TTY), or
# prints it only when TEXT changes (when output is piped to a file).
LAST_LINE=""
draw_bar () {
  local ndone=$1 total=$2 text=$3 width=30 fill
  [ "$total" -gt 0 ] || total=1
  fill=$(( ndone * width / total ))
  local bar
  bar="$(printf '%*s' "$fill" '' | tr ' ' '#')$(printf '%*s' $(( width - fill )) '' | tr ' ' '-')"
  local line
  line=$(printf '[%s] %d/%d  %s  | run %d/%d, sweep %s' "$bar" "$ndone" "$total" "$text" \
         "$RUN_IDX" "$TOTAL_RUNS" "$(fmt_time $(( SECONDS - SWEEP_T0 )))")
  if [ $TTY -eq 1 ]; then
    printf '\r\033[K%s' "${line:0:$(( $(tput cols 2>/dev/null || echo 160) - 1 ))}"
  elif [ "$text" != "$LAST_LINE" ]; then
    echo "$line"; LAST_LINE=$text
  fi
}
end_bar () { [ $TTY -eq 1 ] && printf '\n'; LAST_LINE=""; }

# pkill -f is unreliable under WSL2 (silently misses processes), and a gz sim
# child has been observed to survive SIGINT to the launch process on its own.
# So: try the graceful path, then verify by PID and kill -9 anything still
# alive instead of trusting pkill's exit status.
TEARDOWN_PATTERN='gz sim|ruby.*gz|ros2 launch thesis_rain_sim|run_missions\.py|parameter_bridge|async_slam_toolbox|robot_state_publisher|weather_model_node|rain_model_node|clustering_denoiser|ground_truth_logger|controller_server|planner_server|bt_navigator|behavior_server|lifecycle_manager|velocity_smoother|collision_monitor|smoother_server|route_server|waypoint_follower|opennav_docking|costmap'

teardown () {
  [ -n "$MP" ] && kill "$MP" 2>/dev/null
  [ -n "$LP" ] && kill -INT "$LP" 2>/dev/null && sleep 6
  pkill -f run_missions.py
  pkill -f 'ros2 launch thesis_rain_sim'; pkill -f 'gz sim'; pkill -f 'ruby.*gz'
  pkill -f parameter_bridge; pkill -f async_slam_toolbox; pkill -f robot_state_publisher
  pkill -f weather_model_node; pkill -f rain_model_node
  pkill -f clustering_denoiser; pkill -f ground_truth_logger
  pkill -f 'controller_server|planner_server|bt_navigator|behavior_server|lifecycle_manager|velocity_smoother|collision_monitor|smoother_server|route_server|waypoint_follower|opennav_docking|costmap'
  sleep 4
  # Verification pass: pkill can silently fail under WSL2, so confirm by PID
  # and force-kill anything still alive instead of assuming it worked.
  local stragglers
  stragglers=$(pgrep -f "$TEARDOWN_PATTERN" 2>/dev/null)
  if [ -n "$stragglers" ]; then
    echo " !! teardown: pkill missed PID(s) [$stragglers] - force killing" >&2
    for pid in $stragglers; do kill -9 "$pid" 2>/dev/null; done
    sleep 2
    stragglers=$(pgrep -f "$TEARDOWN_PATTERN" 2>/dev/null)
    [ -n "$stragglers" ] && echo " !! teardown: PID(s) [$stragglers] survived kill -9" >&2
  fi
  LP=""; MP=""
}

on_interrupt () {
  end_bar
  echo " !! Interrupted - tearing down simulation and stopping the sweep"
  teardown
  exit 130
}
trap on_interrupt INT TERM

run_one () {
  local COND=$1 PIPE=$2 R F D
  read -r R F D <<< "$(weather_cli parse "$COND")"
  local PROG="$OUTDIR/.progress_${COND}_${PIPE}"
  RUN_IDX=$(( RUN_IDX + 1 ))
  echo "============================================================"
  echo " RUN $RUN_IDX/$TOTAL_RUNS  world=$WORLD  weather=$COND (rain=$R fog=$F dust=$D)  pipeline=$PIPE  model=$RAIN_MODEL  seed=$SEED"
  echo "============================================================"
  ros2 launch thesis_rain_sim experiment.launch.py \
      world:=$WORLD rain:=$R fog:=$F dust:=$D pipeline:=$PIPE \
      rain_model:=$RAIN_MODEL seed:=$SEED \
      gui:=false rviz:=false nav2:=true slam:=true nav2_delay:=20.0 \
      log_csv:="$LOG_CSV" > "$OUTDIR/launch_${COND}_${PIPE}.log" 2>&1 &
  LP=$!
  # wait for Nav2 to be active (bt_navigator), max ~180s
  local ok=0 t0=$SECONDS
  for i in $(seq 1 60); do
    draw_bar 0 1 "starting sim + Nav2... $(( SECONDS - t0 ))s"
    sleep 3
    if timeout 10 ros2 lifecycle get /bt_navigator 2>/dev/null | grep -q active; then ok=1; break; fi
  done
  end_bar
  if [ $ok -eq 1 ]; then
    echo " Nav2 active - running missions ($LOOPS loops x $N_GOALS goals)"
    rm -f "$PROG"
    ros2 run thesis_rain_sim run_missions.py --ros-args \
      -p use_sim_time:=true -p condition:=$COND -p pipeline:=$PIPE \
      -p localizer:=slam_toolbox -p n_loops:=$LOOPS -p settle_sec:=4.0 \
      -p max_retries:=$MAX_RETRIES -p goal_timeout:=${GOAL_TIMEOUT}.0 \
      -p progress_file:="$PROG" \
      -p "goals:=$GOALS" -p output_csv:="$MIS_CSV" \
      >> "$OUTDIR/missions_${COND}_${PIPE}.log" 2>&1 &
    MP=$!
    t0=$SECONDS
    local phase=starting ndone=0 total=$(( LOOPS * N_GOALS )) detail="starting mission runner"
    while kill -0 "$MP" 2>/dev/null; do
      if [ -f "$PROG" ]; then
        IFS='|' read -r phase ndone total detail < "$PROG"
      fi
      draw_bar "${ndone:-0}" "${total:-1}" "$detail"
      if [ $(( SECONDS - t0 )) -gt $MISSION_WATCHDOG ]; then
        end_bar
        echo " !! Mission runner exceeded ${MISSION_WATCHDOG}s watchdog - killing it"
        kill "$MP" 2>/dev/null; sleep 2; pkill -f run_missions.py
        break
      fi
      sleep 1
    done
    wait "$MP" 2>/dev/null; local rc=$?
    MP=""
    [ -f "$PROG" ] && IFS='|' read -r phase ndone total detail < "$PROG"
    draw_bar "${ndone:-0}" "${total:-1}" "$detail"
    end_bar
    if [ $rc -ne 0 ]; then
      echo " !! run_missions exited with code $rc - see missions_${COND}_${PIPE}.log"
    fi

    echo " Saving SLAM map (for qualitative rain-vs-calm comparison)"
    timeout 25 ros2 run nav2_map_server map_saver_cli \
      -f "$OUTDIR/map_${COND}_${PIPE}" --occ 0.65 --free 0.25 \
      --ros-args -p use_sim_time:=true -p save_map_timeout:=10.0 \
      >> "$OUTDIR/mapsave_${COND}_${PIPE}.log" 2>&1 \
      || echo " !! Map save failed - see mapsave_${COND}_${PIPE}.log"
    rm -f "$PROG"
  else
    echo " !! Nav2 did not become active - skipping missions for $COND/$PIPE"
  fi

  echo " Tearing down"
  teardown
}

for COND in "${CONDS[@]}"; do
  for PIPE in "${PIPES[@]}"; do
    run_one "$COND" "$PIPE"
  done
done

echo "============================================================"
echo " ANALYSIS  (sweep took $(fmt_time $(( SECONDS - SWEEP_T0 ))))"
echo "============================================================"
if [ -f "$MIS_CSV" ]; then
  ros2 run thesis_rain_sim analyse_results.py "$LOG_CSV" \
      --missions "$MIS_CSV" --out "$OUTDIR/summary.csv"
else
  echo " !! No missions.csv (no mission run completed a goal) - skipping mission KPIs"
  ros2 run thesis_rain_sim analyse_results.py "$LOG_CSV" --out "$OUTDIR/summary.csv"
fi
echo
echo "Results in: $OUTDIR"
