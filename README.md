# SPIN: Streamer Parameter Inference with Nested Sampling

Ballistic streamer/infall modeling for streamers around protostars. The code integrates trajectories under stellar gravity and fits them to observed position-position-velocity (PPV) points using dynamic nested sampling.

Fitting scripts are configured through top-level constants; inspect all priors, fixed parameters, data paths, output suffixes, and sampler settings before starting an expensive run.

## Quick Start

```bash
python run_streamer_model.py          # Integrate and visualize one trajectory
python fit_streamer_ns_v2.py    # Current single-lobe dynamic nested sampling
python fit_streamer_ns_combine.py     # Joint blue/red dynamic nested sampling
```

The nested-sampling configurations use many live points and can run for a long time. For development, prefer `python -m py_compile`, imports, and one short trajectory or likelihood evaluation. Do not start a production sampling run as a smoke test.

## Repository Overview

| Path | Purpose |
|------|---------|
| `streamer_ic.py` | Mendoza/Ulrich initial conditions and coordinate conversion |
| `streamer_model.py` | Gravity/drag ODE, RK45 integration, stopping event, azimuth cutoff, tube sampling |
| `run_streamer_model.py` | Single-trajectory configuration and Plotly visualization |
| `fit_streamer_ns_v2.py` | Current single-lobe dynamic nested sampling with optional flux-weighted PPV loss |
| `fit_streamer_ns_combine.py` | Joint blue/red dynamic nested sampling with per-lobe flux weights |
| `visual_orbit.py` | Single- and combined-lobe PPP/PPV visualization |
| `cluster_ns_samples.py` | Nested-sampling posterior clustering |
| `calc-j.py`, `calc-j-ns.py`, `compare-j.py` | Angular-momentum post-processing |
| `plot_corner.py`, `plot_corner_ns.py` | Posterior plotting utilities |
| `estimate_ic.ipynb`, `channel-peak.ipynb` | Initial-condition and PPV-data exploration |
| `legacy/` | Earlier MCMC and nested-sampling scripts retained for reproducibility |
| `docs/superpowers/` | Historical specifications and implementation plans |

Generated `*.h5`, `*.npz`, `*.png`, and `*.html` files are experiment outputs, not source files. Use a distinct `SAVE_SUFFIX` for every configuration to avoid overwriting previous results.

## Model Parameters

The modified Mendoza model combines radial infall with rigid-body rotation (`omega × r`) around a user-defined axis and optional linear drag.

| Parameter | Unit | Description |
|-----------|------|-------------|
| `x`, `y`, `z` | AU | Cartesian starting position; `z` is the line-of-sight coordinate |
| `v_r` | km/s | Initial radial velocity; negative values describe infall |
| `omega` | round/yr | Rotation angular speed; nested-sampling priors may be uniform in `log10(omega)` |
| `theta_axis` | deg | Rotation-axis angle measured from `+y` |
| `phi_axis` | deg | Rotation-axis angle measured from `+z` in the X-Z plane |
| `M` | solar masses | Central stellar mass |
| `alpha` | integration-time unit | Linear-drag damping timescale; larger values mean weaker drag |

Fitting scripts use `PARAM_CONFIG`. Set `is_constant=True` with a `value` for fixed parameters, or `is_constant=False` with `prior_range=[lo, hi]` for free parameters. An entry with `log_uniform=True` interprets its prior bounds in log10 space and passes `10**value` to the trajectory model.

## Coordinate Convention

The angular convention is not standard spherical coordinates:

- `theta` is measured from `+y` toward the X-Z plane.
- `phi` is measured counterclockwise from `+z` in the X-Z plane.

```text
x = -r * sin(theta) * sin(phi)
y =  r * cos(theta)
z =  r * sin(theta) * cos(phi)
```

Use `cartesian_to_spherical()` in `streamer_ic.py` for the inverse conversion. The integrated `vz` component is used as `v_los`.

## Observational PPV Data

The default observational products are located in the parent directory:

```text
../blue-ppvf.npz
../red-ppvf.npz
```

Each file contains equal-length arrays named `x`, `y`, `v`, and `flux`. The points are channel-by-channel dendrogram-leaf centroids, while `flux` is the corresponding integrated leaf flux.

## End-to-End Workflow: Cube to Checked Fit

The fitting step should start only after the spectral cube, extracted peaks, and any region selection have been checked visually. The recommended loop is:

```text
full cube -> velocity subcube -> channel peaks + moment maps
          -> peak overlay check -> optional DS9 region selection
          -> AU/relative-velocity PPV -> nested-sampling fit
          -> moment 0/1 + peaks + fitted-trajectory check
```

The commands below assume the current directory is `lb/model/` and use the `astro` conda environment. Replace the HNCO paths, rest frequency, velocity range, RMS, and region file for the line being analyzed. All processing scripts also support `-h`; CLI arguments override their top-level configuration blocks.

### 1. Cut the line cube

Choose a velocity interval wide enough to include the line and nearby line-free channels:

```bash
python ../cut-subcube.py \
  ../09018-spw2.fits ../data/HNCO/HNCO-cut.fits \
  --restfreq-ghz 219.798274 \
  --velocity-range-kms -10 30
```

Check the printed input/output channel ranges and spectral-axis limits before continuing. Do not use `--overwrite` unless replacing an existing product is intentional.

### 2. Find channel-by-channel peaks and make the first check plots

`prepare-peak.py` masks the cube for moment 0/1, runs a 2-D dendrogram independently in each selected channel, and writes the peak table and diagnostic figures:

```bash
python ../prepare-peak.py \
  ../data/HNCO/HNCO-cut.fits \
  --molname HNCO \
  --restfreq-ghz 219.798274 \
  --rms 8.8e-4 \
  --channel-range 20 40 \
  --moment-mask-sigma 5 \
  --dendro-min-value-sigma 5 \
  --dendro-min-delta-sigma 1 \
  --auto-min-npix \
  --plot-center 76 75 \
  --plot-size 100 140 \
  --output-root ../data
```

This creates, under `../data/HNCO/`:

- `HNCO-prepared-peaks.npz` with `x`, `y`, `v`, `flux`, pixel coordinates, channel indices, and provenance metadata;
- `HNCO-prepared-mom0.fits` and `HNCO-prepared-mom1.fits`;
- `HNCO-prepared-peaks-on-mom01.png`, the main peak-on-moment-map check;
- `HNCO-aoi-average-spectrum.png`, for checking the chosen channel/velocity interval.

Inspect both PNG files. Confirm that the selected channels contain the intended transition, peaks follow real compact emission rather than noise/sidelobes, the moment mask is sensible, and obvious unrelated structures are not entering the fit. If the result is poor, adjust the RMS, channel range, dendrogram thresholds, minimum area, or plot area and regenerate the products before fitting. `prepare-peak-3d.py` is an alternative when a 3-D PPV dendrogram and leaf centroids are more appropriate than independent 2-D channel peaks.

### 3. Select the desired streamer and convert to fitting coordinates

Draw one or more include/exclude regions in DS9 after examining the peak overlay, then apply them with `mask_peak.py`. The current project convention is center pixel `(76, 75)`, distance `2600 pc`, and systemic velocity `10 km/s`:

```bash
python ../mask_peak.py \
  ../data/HNCO/HNCO-prepared-peaks.npz \
  ../data/HNCO/HNCO-prepared-mom0.fits \
  ../HNCO-north.reg \
  --output-npz ../data/HNCO/HNCO-north-masked.npz \
  --output-figure ../data/HNCO/HNCO-north-masked-overlay-au.png \
  --output-au-relative-npz ../data/HNCO/HNCO-north-masked-au-relative-vsys.npz \
  --distance-pc 2600 \
  --center-xy 76 75 \
  --vsys 10
```

The region argument may be omitted to retain all peaks. Always inspect the resulting overlay and the reported input/selected counts. For fitting, use the `*-au-relative-vsys.npz` product: `x` and `y` are offsets in AU from the adopted center, while `v = v_absolute - vsys` in km/s. Repeat this step with separate regions for north/south or blue/red components when fitting lobes separately or jointly.

### 4. Configure and run the fit

Before a production run, edit the fitting script's top-level configuration and verify at least:

- `OBS_DATA` (or `OBS_DATA_BLUE` and `OBS_DATA_RED`) points to the checked `*-au-relative-vsys.npz` file(s);
- `PARAM_CONFIG` has physically justified fixed values and prior ranges;
- `SAVE_SUFFIX` is unique;
- `SIGMA_XY`, `SIGMA_V`, flux-weight settings, trajectory limits, and sampler settings are intentional.

Run cheap checks first, then start the selected fit from `lb/model/` so the existing relative data paths resolve correctly:

```bash
python -m py_compile \
  streamer_ic.py streamer_model.py fit_streamer_ns_v2.py
python fit_streamer_ns_v2.py
```

Use `fit_streamer_ns_combine.py` for a joint two-lobe fit. Production nested sampling can be expensive; a syntax/import or short trajectory check is not a substitute for validating the observational peak selection.

### 5. Plot and inspect the fitted trajectory

The fit writes `results/ns_trajectory<SAVE_SUFFIX>.npz` together with the sampling summary, trace, corner, and interactive best-fit plots. Overlay the final trajectory on the moment maps and the exact PPV points used by the fit:

```bash
python ../plot-traj-au.py \
  ../data/HNCO/HNCO-prepared-mom0.fits \
  ../data/HNCO/HNCO-prepared-mom1.fits \
  --peak-npz ../data/HNCO/HNCO-north-masked-au-relative-vsys.npz \
  --trajectory-npz results/ns_trajectory_hnco_north.npz \
  --distance-pc 2600 \
  --center-xy 76 75 \
  --vsys 10 \
  --output ../data/HNCO/HNCO-north-fit-review.png
```

For multiple lobes, pass all peak files after `--peak-npz` and all matching trajectory files after `--trajectory-npz`. Use `--trajectory-slice-stops STOP ...` only when a documented physical or display reason justifies truncating a trajectory.

Final acceptance should include all of the following checks:

- peaks and trajectory occupy the intended structure in moment 0;
- the peak and trajectory colors agree with the absolute-velocity field in moment 1;
- the trajectory covers the PPV trend without relying on a few high-flux points;
- trace/corner plots and effective-sample/evidence diagnostics do not show an obviously unfinished or boundary-dominated fit;
- center, distance, systemic velocity, input NPZ, priors, and `SAVE_SUFFIX` are recorded consistently.

If any check fails, return to the relevant earlier stage—peak thresholds/channel range, DS9 region, coordinate conversion, or fit priors—and write a new output name rather than silently overwriting the previous experiment.

PPV distances are normalized by default as:

```text
x / SIGMA_XY, y / SIGMA_XY, v / SIGMA_V
SIGMA_XY = 60 AU
SIGMA_V  = 1.331 km/s
```

## Flux-Weighted PPV Loss

The current nested-sampling scripts can weight the observed data-to-model Chamfer term by flux:

```text
loss_dm = sum_i(w_i * d_i**2)
log_likelihood = -0.5 * loss
```

The configuration is:

```python
USE_FLUX_WEIGHTS = True
FLUX_WEIGHT_GAMMA = 1.0
FLUX_CLIP_PERCENTILES = (5.0, 95.0)
```

For each dataset, flux is percentile-clipped, raised to `FLUX_WEIGHT_GAMMA`, and normalized so that the resulting weights have mean one:

```text
w_i = clipped_flux_i**gamma / mean(clipped_flux**gamma)
```

- `USE_FLUX_WEIGHTS=False` disables flux weighting.
- `FLUX_WEIGHT_GAMMA=0` also restores exact equal weighting.
- `gamma=0.5` gives tempered square-root weighting.
- `gamma=1` gives linear flux weighting.

Flux is used here as a geometric importance/reliability weight. The ballistic model does not predict density, excitation, abundance, or radiative transfer, so this must not be interpreted as a physical emission-flux likelihood.

In the combined fit, blue and red weights are normalized independently within each lobe. The joint loss remains the sum over all blue and red points, so the different numbers of observed points still affect the relative lobe contributions.

`fit_streamer_ns_combine.py` also exposes the model-to-data Chamfer term:

```python
USE_MODEL_TO_DATA_LOSS = False
MODEL_TO_DATA_WEIGHT = 1.0
MODEL_TO_DATA_N_SAMPLE = 200
```

The default matches the current v2 single-lobe likelihood and uses only data-to-model distances. Set `USE_MODEL_TO_DATA_LOSS=True` to restore the earlier symmetric Chamfer loss.

## Dynamic Nested Sampling

Both current fitting scripts use `dynesty.DynamicNestedSampler` with multiprocessing, deterministic sampler/resampling seeds, periodic checkpoints, and reproducible equal-weight posterior resampling.

Important settings include:

```text
NLIVE_INIT, NLIVE_BATCH, N_EFFECTIVE, DLOGZ_INIT
N_CPUS, SAMPLER_SEED, RESAMPLE_SEED, CHECKPOINT_EVERY
OBS_DATA or OBS_DATA_BLUE/OBS_DATA_RED
SAVE_SUFFIX
STOPPING_R, AZIMUTH_MAX_DELTA_DEG, T_SPAN, T_EVAL
```

The sample archive records raw/equal-weight samples, `logvol`, `logl`, `logwt`, importance weights, evidence estimates, seeds, and flux/loss configuration. Combined archives additionally save the realized blue and red data weights.

Typical outputs are:

```text
ns_checkpoint<SAVE_SUFFIX>.h5
ns_samples<SAVE_SUFFIX>.npz
ns_summary<SAVE_SUFFIX>.png
ns_trace<SAVE_SUFFIX>.png
ns_corner<SAVE_SUFFIX>.png
ns_bestfit<SAVE_SUFFIX>.html
ns_trajectory<SAVE_SUFFIX>.npz
```

## Validation

Cheap checks should precede any sampling run:

```bash
python -m py_compile streamer_ic.py streamer_model.py
python -m py_compile fit_streamer_ns_v2.py
python -m py_compile fit_streamer_ns_combine.py
```

When changing flux weighting, verify that weights are finite, positive, have mean one, and become all ones with `USE_FLUX_WEIGHTS=False` or `FLUX_WEIGHT_GAMMA=0`.

## Dependencies

- `numpy`, `scipy` — numerical arrays, KDTree queries, and ODE integration
- `astropy` — physical constants and unit conversion
- `dynesty` — dynamic nested sampling
- `emcee` — legacy MCMC workflows
- `matplotlib`, `corner` — diagnostic and posterior plots
- `plotly` — interactive PPP/PPV visualization
