# HiveNavigator — acoustic & sensor analysis of queenright vs. queenless bee colonies

## context

HiveNavigator is an automated hive monitoring platform that detects biologically significant colony events
— queen loss, swarming, parasite pressure — from a continuous stream of acoustic, vibration,
and environmental sensor data.

This analysis focuses on a **controlled queen-removal experiment** conducted in March 2026.
Two hives underwent queen manipulation while the others served as undisturbed controls.
The goal is to characterise how acoustic and vibrational signatures change when a colony loses its queen,
and to build an anomaly detector that can flag queenless states automatically.

![<!-- TODO: photo of the apiary or sensor setup -->](images/apiary_overview.jpg)

---

## the experiment

The experiment spans roughly 10 days (7–16 March 2026).
There are 7 hives with bees and 1 dummy (hive 11, used as sensor baseline).

| hive    | queen status                                    | notes                                              |
|---------|-------------------------------------------------|----------------------------------------------------|
| hive 03 | queenless from 12 March onward                  | queen removed on 12 March                          |
| hive 04 | queenless 9–12 March, queen introduced 12 March | queen was absent, then a queen was introduced      |
| others  | queenright (control)                             | undisturbed throughout                              |
| hive 11 | dummy                                           | no bees — used to identify structural sensor noise  |

---

## data sources

### audio files

Mono FLAC recordings at 16 000 hertz sample rate, two files per hour per hive.
Each file covers approximately 30 minutes of continuous in-hive audio.

### accelerometer

Sampled approximately every 60 seconds.
Seven columns: timestamp, three main fast Fourier transform frequencies (f1, f2, f3)
and their corresponding magnitudes (m1, m2, m3).

### environmental sensors

Sampled every 10 minutes.
Temperature, relative humidity, CO₂ concentration, accelerometer orientation, and Wi-Fi signal strength.

---

## audio feature extraction pipeline

For each audio file, interpretable spectral features are computed over short analysis frames
(1–2 second Hann windows, 50% overlap) and averaged into a single feature vector representing that hour.

### preprocessing

Audio is bandpass-filtered (6th-order Butterworth, 100–2 000 hertz) to remove
sub-bass rumble and high-frequency noise before feature extraction.

### core spectral features

- **mel-frequency cepstral coefficients** — 13 coefficients capturing the coarse spectral envelope
- **spectral centroid** — frequency-weighted centre of mass of the spectrum
- **spectral bandwidth** — spread of energy around the centroid
- **spectral rolloff** — frequency below which a given fraction of total energy is concentrated
- **spectral flatness** — ratio of geometric to arithmetic mean of the power spectrum; high values indicate noise-like activity
- **chroma features** — 12-bin pitch-class energy distribution; picks up tonal piping
- **zero-crossing rate** — proxy for noisiness and high-frequency content
- **root-mean-square energy** — overall loudness envelope

### modulation spectrogram

The modulation spectrogram captures how energy in each frequency band fluctuates over time:

1. compute the short-time Fourier transform to obtain an ordinary spectrogram S(t, f)
2. apply a second fast Fourier transform along the time axis of each frequency row → M(ω_m, f)
3. average M across time frames for each hour

The modulation rates most informative for bee behaviour are typically 1–30 hertz
(wing-beat modulation around the 250 hertz carrier, queen and worker piping, fanning).

### passband features

Features are also computed per frequency band (low, middle, high passbands)
to capture energy distribution across biologically relevant ranges.

### aggregation

Frame-level features are aggregated per hour time slice (e.g. 11–12, 12–13, 13–14)
to produce one feature vector per hive per hour.

![<!-- TODO: spectrogram comparison queenright vs queenless -->](images/spectrogram_comparison.png)

---

## accelerometry analysis

### data overview

The accelerometer records the three main fast Fourier transform peak frequencies and their magnitudes
every ~60 seconds across 11 hives.

### cleaning steps

1. **kept only the first peak** — peaks 2 and 3 are echoes visible on spectrograms and do not bring additional information
2. **removed structural harmonics** — three narrow fixed frequency bins (~110, ~220, ~330 hertz) identical across all hives including the dummy; assumed to be structural resonance of the sensor or hive body
3. **magnitude threshold of 6** — filters out diffuse noise; keeps only genuine vibration patterns
4. **restricted to daytime hours (8:00–18:00)** — night carries little vibrational information

![<!-- TODO: accelerometry frequency overview before cleaning -->](images/accelerometry_raw_overview.png)

![<!-- TODO: spectrogram by peak rank showing echo pattern -->](images/accelerometry_peak_rank_spectrograms.png)

![<!-- TODO: histogram before and after harmonic removal -->](images/accelerometry_harmonic_removal.png)

### indicator: daily in-band peak ratio

The explored indicator is the **daily in-band peak ratio**: the fraction of peaks falling in the 50–100 hertz band
versus all frequencies.

The 50–100 hertz band is where exploitable hives concentrate vibrational energy.
The higher band (200+ hertz) is more diffuse and less sharply bounded than the literature suggests.
Comparing 50–100 hertz against everything else captures the balance between these two regimes.

**observations:**

- hive 04 starts with a particularly low ratio and begins to rise around 12 March
- hive 03 follows the opposite trend
- the two curves crossing coincides with the queen being transferred between them

**conclusion:**
this does not appear to be a strong standalone indicator of queenright/queenless state,
neither for intra-hive monitoring nor inter-hive comparison.
However, it may be a useful contributor alongside audio features in a multimodal approach.

![<!-- TODO: daily in-band peak ratio plot with queen state hints -->](images/daily_in_band_peak_ratio.png)

---

## environmental sensors

Temperature and humidity are read from the SHT sensor inside each hive.
These values are merged with the audio feature dataframe by timestamp
and included as context features for anomaly detection.

![<!-- TODO: temperature and humidity over time for a representative hive -->](images/temperature_humidity_timeseries.png)

---

## feature merging

All per-hive audio feature dataframes are concatenated, then left-joined with the ambient sensor data
(temperature, humidity) on timestamp. The merged dataframe is the input to anomaly detection.

Each row represents one hour of one hive and contains:
timestamp, hive identifier, time slice, queen state label, all audio features, temperature, and humidity.

---

## anomaly detection

### method

A **one-class support vector machine** is trained on queenright data only (all hives, all hours).
The model learns the "normal" feature envelope; queenless observations are scored as novelty.

**Per-time-slice scoring:** one scaler and one detector are fitted per hour slice,
so the model does not confuse day/night variation with anomaly.

**Feature selection:** features are filtered to keep only those whose z-score
diverges in opposite directions between queenright and queenless data.

### focused time slices

Activity is concentrated during daylight.
The analysis focuses on midday slices (11–12, 12–13, 13–14) where bee activity is highest
and the signal-to-noise ratio is best.

### evaluation

- **anomaly score histograms** — visual check that queenless scores shift left (more anomalous)
- **Mann–Whitney U test + area under the receiver operating characteristic curve** — non-parametric measure of how separable the two populations are
- **Cohen's d** — effect size in pooled standard deviations

![<!-- TODO: discrimination figure for all queenless vs queenright -->](images/discrimination_all_queenless.png)

![<!-- TODO: discrimination figure hive 03 -->](images/discrimination_hive_03.png)

![<!-- TODO: discrimination figure hive 04 -->](images/discrimination_hive_04.png)

### mosaic visualisation

The mosaic heatmap provides a detailed view of the anomaly:

- **rows** = hourly observations ordered chronologically
- **columns** = features, z-scored against the queenright baseline
- **left strip** = anomaly score (darker = more anomalous, red border = bottom 1st percentile)
- **baseline panel** = the same time window aggregated across queenright hives (mean, worst, furthest from centroid, or single hive — selectable)
- **column ordering** = features reordered by similarity for visual clustering

![<!-- TODO: mosaic heatmap for queenless hive 04 -->](images/mosaic_hive_04.png)

![<!-- TODO: mosaic heatmap for queenless hive 03 -->](images/mosaic_hive_03.png)

---

## findings

- **hive 04** queenless period is well separated from the queenright baseline in the anomaly score distribution.
  The mosaic heatmap highlights which features deviate most and at which hours.
- **hive 03** queenless period shows weaker separation, if any.
  This may reflect the colony's slower response to queen removal,
  the influence of low temperatures on bee activity, or simply a less pronounced acoustic shift.
- The **mosaic heatmap** makes it possible to identify both the temporal extent of the anomaly
  (which hours diverge) and the spectral nature of the deviation (which features shift).
- **Accelerometry alone** is not a reliable queenlessness indicator,
  but the daily in-band peak ratio shows suggestive trends that align with queen transfer timing.
- **Combining audio features with environmental data** (temperature, humidity) in a multimodal approach
  provides the most informative anomaly detection.

---

## interactive dashboard

A Streamlit application allows interactive exploration of the anomaly detection results.
The user can select queenless hives, choose features, pick an anomaly detector,
and view discrimination plots and z-score mosaics interactively.

See [`app.py`](app.py) for the dashboard source.

---

## repository structure
```
.
├── app.py                          # streamlit dashboard
├── README.md                       # this report
├── python/
│   └── notebooks/
│       ├── paths.py                # file path conventions
│       ├── dataframes.py           # dataframe loading and z-scoring
│       ├── anomalies.py            # anomaly scoring, discrimination, mosaic
│       ├── features.py             # audio feature extraction
│       ├── passbands.py            # frequency band definitions
│       ├── timestamps_and_frames.py # audio file → (timestamp, samples) stream
│       ├── timestamps_and_accelerometries.py # accelerometry pipeline
│       ├── normalization.py        # z-score normalisation
│       ├── aggregations.py         # per-time-slice aggregation
│       ├── times.py                # time slice tagging, date ranges
│       ├── plot.py                 # matplotlib/plotly plotting utilities
│       ├── sensors.py              # sensor csv parsing
│       ├── audio_data_processing.py # end-to-end feature extraction notebook
│       ├── accelerometry.py        # accelerometry analysis notebook
│       ├── merging_features.py     # feature merging notebook
│       └── anomaly_detection.py    # anomaly detection notebook
├── data/
│   ├── Audio/                      # raw FLAC files per hive
│   ├── Sensors/                    # accelerometer + sensor CSVs
│   └── features/                   # extracted feature CSVs
└── images/                         # figures for this report
```

---

## how to run
```bash
pip install librosa soundfile numpy scipy pandas scikit-learn matplotlib plotly streamlit pipe
```

**feature extraction** (run notebooks in order):
1. `audio_data_processing.py` — extract audio features per hive
2. `merging_features.py` — merge all hives + ambient sensors into one dataframe

**analysis:**
3. `accelerometry.py` — accelerometry exploration
4. `anomaly_detection.py` — anomaly detection and evaluation

**dashboard:**
```bash
streamlit run app.py
```
