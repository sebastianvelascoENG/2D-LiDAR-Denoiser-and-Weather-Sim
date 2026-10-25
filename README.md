# 2D LiDAR Denoiser and Weather Sim

A ROS 2 simulation framework for measuring how **rain, fog and dust** degrade a
2D LiDAR, and whether a **clustering / keyframe denoiser** (Hu et al. 2022)
recovers autonomous navigation performance.

A TurtleBot3 Waffle, with its LiDAR retuned to an **RPLiDAR C1**, drives fixed
missions in Gazebo while a physics-based weather model corrupts its scans in
real time. SLAM and Nav2 run on either the raw or the denoised scan, and every
run is scored against a drift-free ground-truth pose.

**Stack:** ROS 2 Jazzy · Gazebo Sim (Harmonic) · slam_toolbox · Nav2 (DWB) · Python / NumPy

**ROS package name:** `thesis_rain_sim`

**Highlights**
- Four interchangeable rain models, three of them implemented from published
  papers (Goodin 2019, Alavi 2025, LISA / Kilic 2021) with every coefficient
  traced to its source.
- Fog and dust (Hahner 2021, Rasshofer 2011, Alavi 2025): rain, fog and dust
  can be switched on independently and in any combination, and all of them
  act on one simulated sensor.
- Per-beam ground-truth labels, so the denoiser can be scored directly.
- Unattended batch experiments with a progress bar, watchdogs, retries and
  clean teardown, plus automatic statistics.
- Over 100 unit tests covering the weather physics.

---

## Pipeline

```
gz gpu_lidar ──/scan_clean──▶ weather_model_node ──/scan──▶ clustering_denoiser_node ──/scan_denoised──▶
                                   │                                                   │
                               (rain/fog/dust +                                   (low-intensity
                                modelled intensity)                                cluster removal)
                                   │                                                   │
                     pipeline:=raw ┘                          pipeline:=denoised ──────┘
                                   ▼
                       slam_toolbox  +  Nav2   ──▶  ground_truth_logger  ──▶  CSV
                                                    (gt pose vs SLAM est, scan quality)
```

`/ground_truth` is a drift-free pose from a Gazebo `OdometryPublisher` on the
robot. It is used only for scoring, never for navigation. Costmaps and the
collision monitor read the selected pipeline's scan, so weather clutter affects
obstacle avoidance as well as SLAM.

---

## What's included

```
thesis_rain_sim/
├── thesis_rain_sim/          # pure-NumPy physics library (no ROS dependency)
│   ├── rain_models.py        #   legacy, goodin2019, alavi2025, hybrid rain models
│   ├── fog_dust.py           #   fog / dust extinction, backscatter and soft-echo model
│   ├── dust_geometry.py      #   2D ray–circle / ray–box chord lengths for dust clouds
│   └── weather.py            #   weather presets and run labels
├── scripts/                  # ROS 2 nodes and experiment tooling
├── launch/
│   ├── experiment.launch.py  #   full stack: sim + weather + denoiser + SLAM + Nav2 + logger
│   ├── sim.launch.py         #   Gazebo world, robot and ROS–gz bridge
│   └── nav2.launch.py        #   Nav2 bringup, remapped to the selected pipeline's scan
├── config/
│   ├── weather_params.yaml   #   rain / fog / dust model parameters
│   ├── dust_clouds.yaml      #   dust cloud placements per world
│   ├── denoiser_params.yaml  #   clustering denoiser parameters
│   ├── slam.yaml             #   slam_toolbox
│   ├── nav2_waffle.yaml      #   Nav2 (DWB), costmaps, collision monitor
│   └── bridge.yaml           #   ros_gz_bridge topics
├── worlds/                   # rain_room.sdf (enclosed), rain_yard.sdf (open yard)
├── models/tb3_waffle_c1/     # TurtleBot3 Waffle with an RPLiDAR C1 (500 beams, 12 m, 10 Hz)
├── behavior_trees/           # Nav2 BT: replan if the path becomes invalid
├── rviz/experiment.rviz      # RViz layout (raw vs denoised scans, map, plan)
└── test/                     # pytest suites + a recorded rain_yard scan
```

### Nodes and scripts

| executable | role |
|---|---|
| `weather_model_node.py` | `/scan_clean` → `/scan`. Applies rain, fog and dust, models return intensity, and publishes per-beam labels on `/scan_rain_labels` and its state on `/weather/state`. Every effect can be changed live with `ros2 param set`. |
| `clustering_denoiser_node.py` | `/scan` → `/scan_denoised`. Keyframe descriptor, annular region segmentation, angular clustering and a cross-region merge, then drops clusters below an intensity threshold or a minimum point count. |
| `ground_truth_logger.py` | Logs position / yaw error (SLAM vs ground truth) and valid-beam counts for all three scans to CSV. |
| `run_missions.py` | Drives a fixed Nav2 goal set and logs result, duration, distance and recoveries per goal. Every ROS call is time-bounded, and goals time out and are retried. |
| `analyse_results.py` | Per-condition summary statistics, raw-vs-denoised and weather-vs-calm comparisons, and mission KPIs (NumPy + SciPy). |
| `run_sweep.sh` | Runs a full {weathers} × {raw, denoised} batch for one world, then the analysis. |
| `overnight.sh` | Runs `run_sweep.sh` back to back with a new seed each round, until stopped. |
| `plot_rain_models.py` | Offline (not a node): plots range and valid-beam curves vs rain rate for every model, plus summary tables. |

---

## Weather models

### Rain (`rain_model:=`)

| model | source | behaviour |
|---|---|---|
| `hybrid` *(default)* | LISA, Kilic et al. 2021 (Marshall–Palmer drops, extinction, range noise) + the Alavi 2025 sensor model | Drops are placed in each beam's cone. Gives slight range loss, rare near-range false returns from individual drops, and a few mm of rain range noise. |
| `goodin2019` | Goodin et al., *Electronics* 8(1):89, 2019 | Strong attenuation, so range loss plus range noise. No false returns, by design. |
| `alavi2025` | Alavi et al., *Appl. Sci.* 15(15):8191, 2025 | Weak extinction. Stochastic volume backscatter gives occasional near-range false returns. |
| `legacy` | Original ad-hoc model | Heavy clutter and random dropout. Kept as a baseline. |

All physical models share one sensor: a Lambertian hard-target return
P = (ρ/π)/z² · e^(−2αz), and a detection threshold derived from the C1 data
sheet (12 m at 70 % reflectivity). A beam below the threshold is published as
`+inf`.

Rain presets: `calm` / `light` / `heavy` = 0 / 6 / 25 mm/h. Any other rate can
be set with `rain_rate_mm_h`.

| model | max range of a 70 % target at 0 / 6 / 25 mm/h | false returns per scan at 6 / 25 mm/h |
|---|---|---|
| legacy | 11.95 / 11.94 / 11.94 m | 11.7 / 50.0 |
| goodin2019 | 11.95 / 9.17 / 7.26 m | 0 / 0 |
| alavi2025 | 11.95 / 11.88 / 11.69 m | 0.0 / 7.7 |
| hybrid | 11.95 / 11.83 / 11.60 m | 0.19 / 0.58 |

### Fog and dust (`fog:=`, `dust:=`)

| effect | levels | meaning |
|---|---|---|
| `fog` | off / light / heavy | Visibility 100 / 50 m, fills all space |
| `dust` | off / light / heavy | Visibility 20 / 5 m inside the dust clouds of `config/dust_clouds.yaml` |

- Fog and dust extinction multiplies every return, including the rain echoes.
- The volume echo is computed by integrating the pulse over the medium (Hahner
  et al. 2021, Eq. 18). It competes with the hard target and any rain echoes,
  and the strongest echo above threshold is published (single-return sensor).
- Fog and dust draw from a separate random stream, so turning them on does not
  change the rain realisation for a given seed.
- Fog and dust require one of the physical rain models (not `legacy`).

### Ground-truth labels (`/scan_rain_labels`)

A `LaserScan` stamped like the matching `/scan`. Its `ranges` are the clean
ranges and its `intensities` hold one label per beam:

| code | meaning |
|---|---|
| 0 | real return, sensor noise only |
| 1 | real return with weather range noise |
| 2 | real surface lost to the weather |
| 3 | false return from a rain droplet |
| 4 | no surface and no weather echo |
| 5 | false return from fog backscatter |
| 6 | false return from dust backscatter |

To score the denoiser, positives are labels 3, 5 and 6 and negatives are
labels 0–1.

---

## Requirements

- Ubuntu 24.04 (native or WSL2)
- ROS 2 Jazzy, Gazebo Sim Harmonic, `ros_gz`
- `turtlebot3` packages, `slam_toolbox`, `navigation2`, `nav2_bringup`
- Python 3.12 with `numpy`, `scipy`, `matplotlib`, `pytest`

```bash
sudo apt install ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-slam-toolbox \
                 ros-jazzy-turtlebot3* ros-jazzy-ros-gz ros-jazzy-tf-transformations \
                 python3-numpy python3-scipy python3-matplotlib python3-pytest
```

## Build

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/sebastianvelascoENG/2D-LiDAR-Denoiser-and-Weather-Sim.git thesis_rain_sim
cd ~/ros2_ws
chmod +x src/thesis_rain_sim/scripts/*.py src/thesis_rain_sim/scripts/*.sh
colcon build --packages-select thesis_rain_sim
source install/setup.bash
export TURTLEBOT3_MODEL=waffle
```

The package is built without `--symlink-install`, so rebuild after editing a
script. If you switch between the two modes, delete
`~/ros2_ws/build/thesis_rain_sim` first.

## Test

```bash
cd ~/ros2_ws/src/thesis_rain_sim && python3 -m pytest test/ -v
# or through colcon:
colcon test --packages-select thesis_rain_sim && colcon test-result --verbose
```

The suites check the published reference values for each model, the C1
detection threshold, seed reproducibility, bit-for-bit equivalence of the
`legacy` model, the fog-preset safety margin, and the batched dust integration
against the per-beam reference.

---

## Running

### 1. Interactive: watch one condition in Gazebo and RViz

```bash
ros2 launch thesis_rain_sim experiment.launch.py \
    world:=rain_room rain:=heavy pipeline:=denoised \
    nav2:=true rviz:=true gui:=true
```

Any combination of rain, fog and dust works:

```bash
ros2 launch thesis_rain_sim experiment.launch.py world:=rain_yard \
    rain:=calm fog:=heavy dust:=light pipeline:=raw nav2:=true

ros2 param set /weather_model_node dust off     # change effects while it runs
```

In RViz, white points are `/scan` (with weather) and green points are
`/scan_denoised`. Send goals with the **Nav2 Goal** tool.

### 2. Headless single run with logging

```bash
ros2 launch thesis_rain_sim experiment.launch.py \
    world:=rain_yard rain:=light pipeline:=raw \
    nav2:=true gui:=false rviz:=false log_csv:=~/run1.csv

# in a second terminal, once Nav2 is active:
ros2 run thesis_rain_sim run_missions.py --ros-args \
    -p use_sim_time:=true -p condition:=light -p pipeline:=raw \
    -p localizer:=slam_toolbox -p output_csv:=~/missions.csv

# summarise:
ros2 run thesis_rain_sim analyse_results.py ~/run1.csv --missions ~/missions.csv
```

### 3. Full batch experiment

```bash
ros2 run thesis_rain_sim run_sweep.sh rain_yard
```

By default this runs calm, light and heavy rain, each on both the raw and the
denoised pipeline, and then runs the analysis. You can choose the weathers in
two ways. `WEATHER` takes an explicit list of labels. Without it, the sweep
runs every combination of `RAIN_LEVELS` × `FOG_LEVELS` × `DUST_LEVELS`:

```bash
RAIN_MODEL=alavi2025 SEED=2 ros2 run thesis_rain_sim run_sweep.sh rain_yard
RAIN_LEVELS=calm FOG_LEVELS="off light heavy" ros2 run thesis_rain_sim run_sweep.sh rain_yard
WEATHER="calm heavy fog_heavy dust_heavy rain_heavy+fog_heavy+dust_heavy" \
    ros2 run thesis_rain_sim run_sweep.sh rain_yard
```

Each run shows a live status line:

```
 RUN 1/6  world=rain_yard  weather=calm (rain=calm fog=off dust=off)  pipeline=raw  model=hybrid  seed=1
[###############---------------] 4/8  loop 2/2 goal 1/4 (try 2) | 4.1 m left | 35s  | run 1/6, sweep 22:44
```

Labels are validated before anything starts. A watchdog kills a stuck mission
runner, and teardown force-kills any leftover processes. **Ctrl-C** stops the
whole stack cleanly.

### 4. Overnight: repeated sweeps with new seeds

```bash
ros2 run thesis_rain_sim overnight.sh            # default world: rain_yard
```

Runs `hybrid` sweeps with seed 1, 2, 3, ... until stopped. By default each
sweep covers every condition on its own, then all three together (light and
heavy): 9 weathers × 2 pipelines = 18 runs.

- To stop after the current sweep: `touch ~/rain_sim_results/overnight_*/STOP`
- To stop immediately: Ctrl-C or `kill -TERM <pid>`
- Overrides: `WEATHER`, `FIRST_SEED`, `MISSION_LOOPS`, `MAX_BAD`
- It refuses to start while another `gz sim` is running, and stops by itself
  after `MAX_BAD` sweeps in a row with no successful goal.

> **Run one simulation at a time.** Every run shares one ROS domain, so a second
> simulation publishes a second `/clock` and corrupts both runs' TF and data.
> Check first with `pgrep -af "gz sim"`.

### 5. Offline model curves

```bash
ros2 run thesis_rain_sim plot_rain_models.py \
    --scan src/thesis_rain_sim/test/data/rain_yard_scan_clean.npy
```

Writes plots and the `model_summary.csv` / `hybrid_divergence.csv` tables to
`~/rain_sim_results/rain_model_curves/`.

---

## Output

Each sweep writes `~/rain_sim_results/<world>_YYYYMMDD_HHMM/`:

| file | contents |
|---|---|
| `nav_error_log.csv` | 2 Hz log: condition, pipeline, rain / fog / dust levels, position and yaw error, valid beams per scan |
| `missions.csv` | One row per goal: result (`SUCCEEDED` / `FAILED` / `CANCELED` / `TIMEOUT` / `NOT_SENT`), duration, distance, recoveries |
| `summary.csv` | Output of `analyse_results.py` |
| `map_*.pgm` / `.yaml` | SLAM map saved at the end of each run |
| `sweep_info.txt` | World, model, seed, start time and git commit |
| `launch_*.log`, `missions_*.log` | Per-run logs from the stack and the mission runner (retries are listed here) |

An overnight run also writes `~/rain_sim_results/overnight_YYYYMMDD_HHMM/`,
which holds `index.csv` (one row per sweep, with goal success counts) and
`overnight.log`.

---

## Configuration

### Launch arguments (`experiment.launch.py`)

| arg | values (default first) |
|---|---|
| `world` | `rain_room` \| `rain_yard` |
| `rain` | `calm` \| `light` \| `heavy` |
| `fog`, `dust` | `off` \| `light` \| `heavy` |
| `fog_visibility_m`, `dust_visibility_m` | override the preset visibility (m) |
| `dust_clouds` | cloud set from `dust_clouds.yaml` (default: named after the world) |
| `rain_model` | `hybrid` \| `goodin2019` \| `alavi2025` \| `legacy` |
| `seed` | integer, default `1` |
| `pipeline` | `raw` \| `denoised` |
| `slam`, `nav2`, `rviz`, `gui` | `true` \| `false` |
| `nav2_delay` | seconds to wait before starting Nav2 (default 15) |
| `log_csv` | path for the ground-truth logger |
| `record_bag`, `bag_path` | record a rosbag of the run |

### Mission goals

| world | goals (x, y), visited in order and looped `MISSION_LOOPS` times |
|---|---|
| `rain_yard` | (6, 6) → (−6, 9) → (−6, −6) → (6, −6) |
| `rain_room` | (3.5, 3.5) → (−2.5, 2.5) → (−2.5, −2.5) → (2.5, −2.5) |

Dust clouds in each world sit on the mission routes, so the robot drives
through dust on every lap.

### Tuning

- **Weather:** `config/weather_params.yaml`, grouped by model (shared sensor
  constants, then the goodin2019, alavi2025, hybrid and fog / dust parameters).
- **Denoiser:** `config/denoiser_params.yaml` (`intensity_threshold`,
  `num_regions`, `angle_threshold_mult`, `min_cluster_points`, `require_both`).
- **Navigation:** `config/nav2_waffle.yaml`. The robot radius is 0.27 m,
  inflation is 0.7 m, the global costmap is a 40 × 40 m rolling window, and the
  collision monitor uses approach + slow zones (no hard stop polygon, so the
  robot can always back away from an obstacle).
- **SLAM:** `config/slam.yaml`.

### Troubleshooting

- If `bt_navigator` does not reach `active`, increase `nav2_delay`. The
  simulation can run below real time on a laptop.
- If `ros2 run` reports `No executable found`, the scripts lost their `+x` bit
  (this happens when they are edited from Windows). Re-run the `chmod` from
  **Build**, then rebuild.
- Gazebo's `gz_frame_id` SDF warnings are harmless.

---

## References

- Goodin, C. et al. "Predicting the Influence of Rain on LIDAR in ADAS." *Electronics* 8(1):89, 2019.
- Alavi, S. et al. *Applied Sciences* 15(15):8191, 2025.
- Kilic, V. et al. "Lidar Light Scattering Augmentation (LISA)." arXiv:2107.07004, 2021.
- Hahner, M. et al. "Fog Simulation on Real LiDAR Point Clouds." ICCV 2021.
- Rasshofer, R. H. et al. "Influences of weather phenomena on automotive laser radar systems." *Adv. Radio Sci.* 9, 2011.
- Marshall, J. S. & Palmer, W. M. "The distribution of raindrops with size." *J. Meteor.* 5:165, 1948.
- Kim, I. I., McArthur, B. & Korevaar, E. "Comparison of laser beam propagation at 785 nm and 1550 nm in fog and haze." SPIE 4214, 2001.
- Hu et al. 2022. Clustering / keyframe LiDAR denoising.

## License

MIT. See [`LICENSE`](LICENSE).
