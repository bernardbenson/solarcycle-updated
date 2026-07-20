# Problem Statement — Solar Cycle Amplitude Forecasting

## 1. What is being attempted

The goal is to forecast an entire solar cycle before it happens: given the
sunspot record up to a forecast origin at or near solar minimum, produce the
next 132 months (one nominal 11-year cycle) of monthly sunspot number, together
with an uncertainty band that means what it says.

Two quantities carry almost all of the practical value:

- **Peak amplitude** — how strong the coming cycle will be.
- **Peak timing** — when, in months from minimum, that maximum arrives.

The full 132-month curve is the delivery format, but it is largely a rendering
of those two numbers plus a broadly stereotyped rise-and-fall shape. A forecast
that gets the curve's texture right and the amplitude wrong is not useful; the
converse largely is.

The forecast must be issued *early* — from a cycle minimum, looking roughly a
decade ahead — because that is when the decisions it informs are made. Cycle
amplitude sets the multi-year backdrop for thermospheric density and therefore
satellite drag and orbital lifetime, for HF radio propagation and GNSS
positioning error, for radiation exposure on polar aviation routes and crewed
missions, and for the background rate of geomagnetically induced currents in
power grids. Mission planners, constellation operators, and grid planners need
the envelope years in advance; a nowcast is too late to be actionable.

This work is a re-examination of Benson et al. (2020), *Forecasting Solar Cycle
25 Using Deep Neural Networks* (Solar Physics 295:65). Solar Cycle 25 has since
been observed, which makes the earlier prediction scoreable, and it did not
score well: the observed smoothed peak reached roughly **156**, against a
predicted **106 ± 19.75**. The observation fell well outside the stated
interval. Understanding *why* — and whether the failure was in the modeling, the
data, or the evaluation — is the motivating question.

## 2. Why this is hard

The central obstacle is not noise, nonlinearity, or missing physics. It is
sample size, and it is disguised.

The sunspot record looks long. Monthly observations run continuously from 1749,
which is more than three thousand monthly values, and daily observations run
from 1818. That abundance is an illusion for this task. **The prediction unit is
the cycle, not the month.** A forecast consumes a stretch of history and emits
one cycle; each historical cycle supplies exactly one training example. The
record contains on the order of **25 complete cycles**, and once a portion is
reserved for honest testing, the effective training set is roughly **20
examples**.

The illusion is quantifiable. Sliding a 528-month input window forward one month
at a time across the monthly record yields on the order of **2,560 input/target
pairs** — a number large enough to look like a respectable dataset, and one that
consists almost entirely of near-duplicates. Consecutive pairs differ by a
single month at each end. The independent information content is unchanged: it
is still about twenty cycles.

Everything difficult about this problem follows from that number:

- A model that emits 132 free monthly values is fitting a 132-dimensional output
  from ~20 observations. There is no regime in which that is identified from
  data alone; whatever structure appears in the output has to come from a prior,
  either an explicit one or an accidental one.
- Evaluation is fragile in a specific way. Scoring on monthly points implies
  thousands of test observations and produces error bars that are far too
  narrow, because adjacent months are nearly perfectly correlated. Splits
  constructed by index rather than by cycle almost always leave train and test
  *target windows* overlapping, and the resulting numbers measure interpolation,
  not forecasting.

The Cycle 25 miss illustrates the second point concretely. The 2020 paper
reported an RMSE of 2.93 — a figure that cannot be reproduced under a split
where the training and validation targets are disjoint, and which reflects a
window overlap rather than forecasting skill. The corrective framing adopted
here is a floor: **climatology — simply predicting the average of all previous
cycles — achieves an RMSE of about 38 in raw sunspot-number units.** Any
reported error meaningfully below that should be treated as suspect until the
split is described in full. Most of the apparent progress in this literature
lives in the gap between those two numbers.

A secondary difficulty is that accuracy and honesty about uncertainty are
separate properties. A forecast can match the best available method on error
while assigning far too much confidence to its central estimate. Both must be
reported, and the 2020 miss was as much an interval failure as a point-estimate
failure.

## 3. Data available

All data are public, observational, and long-baseline. Nothing proprietary and
nothing simulated is involved.

| Dataset | Coverage | Cadence | Role |
|---|---|---|---|
| Sunspot number, SILSO version 2.0 | 1749–present (monthly); 1818–present (daily) | monthly mean, daily | Primary target series |
| Total sunspot area | 1874–present | daily, aggregated monthly | Independent measure of the same physical activity; cross-consistency check |
| Kp / Ap geomagnetic indices | 1932–present | daily | Candidate precursor and exogenous input |
| F10.7 cm solar radio flux | 1947–present | daily | Auxiliary activity index (see caveat below) |

The working table assembled from these sources is a daily merged record of
roughly **81,700 rows spanning 1818-01-01 through 2026-06-30**, carrying, per
day, the sunspot number, its reported standard deviation, the number of
contributing observations, a provisional flag, and — where available — the daily
Kp sum and Ap average. It is also the one data file kept under version control;
everything else is regenerable.

Coverage inside that table is uneven in ways that matter. About **4% of days
carry no sunspot number at all**, and those absences are concentrated in the
1818–1849 stretch where the observing network was thinnest. Roughly **41,600
days — every day before 1932 — carry no geomagnetic value**, which is simply the
start date of the Kp/Ap record. And F10.7, though nominally available from 1947,
is **absent from the current assembled table entirely**: the retrieval produced
no usable columns on the most recent regeneration, so in practice the working
data are sunspot number plus geomagnetic indices. F10.7 should be treated as a
source the pipeline knows how to reach rather than as data currently in hand.

A separate, much larger derived table also exists from an earlier generation of
this work — roughly 131 daily columns of lags, rolling statistics, spectral
features, volatility measures, and activity-regime indicators. It is retained
for reference. Its existence is itself a finding: heavy feature engineering at
daily resolution does not address a problem whose scarcity is at cycle
resolution.

Beyond the observed series, the record itself supports a set of derived
quantities that require no external data and that turn out to matter more than
any of the auxiliary indices:

- The **13-month smoothed sunspot number**, the community convention against
  which cycle peak amplitude and timing are defined.
- **Cycle minima**, and from them **cycle length** measured minimum to minimum.
- **Depth of minimum** — how low activity actually fell between cycles.
- **Activity level in the roughly three years preceding a minimum**, and the
  minimum level expressed as a fraction of the preceding cycle's peak.

Total on-disk footprint is modest and fully regenerable; the working data
directory is not version-controlled.

## 4. Sources

| Provider | Series |
|---|---|
| SILSO / World Data Center, Royal Observatory of Belgium | Daily sunspot number (`sidc.be/SILSO/INFO/sndtotcsv.php`) and monthly mean sunspot number, version 2.0 (`sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv`) |
| Hathaway / Upton active-region database, solarcyclescience.com | Daily total sunspot area (`solarcyclescience.com/AR_Database/daily_area.txt`) |
| GFZ Potsdam | Kp and Ap geomagnetic indices, 1932–present |
| CelesTrak | F10.7 cm solar radio flux, 1947–present (originally Penticton/DRAO) |

One property of this source set is deliberate and worth stating: **every series
is retrieved unsmoothed, at its native cadence, wherever the provider offers
it.** Smoothing is a downstream analytical choice, not an inherited property of
the input. This matters because the 13-month smoothed series that the field
conventionally quotes is a *centered* average — it uses future months — and a
series that arrives pre-smoothed carries look-ahead contamination that cannot be
undone.

## 5. Hypotheses

The project is organized around four claims.

**H1 — Amplitude information lives in precursors, not in the shape of the
history.** The strength of the coming cycle is encoded in the *conditions at and
before the preceding minimum* — how long the previous cycle ran, how deep the
minimum went, how much activity persisted in the years approaching it, and how
the minimum level compares to the previous peak — rather than in the detailed
morphology of the preceding sunspot curve. If true, a model given only raw
history has been handed the wrong representation, and no amount of capacity will
recover what is not there.

**H2 — Cycle length anti-correlates with the amplitude of the following cycle.**
Long cycles are followed by weak ones. This is derivable from the sunspot record
alone and serves as a proxy for the Hale-cycle terminator separation described
by McIntosh et al. (2023), who report a correlation of roughly **r = −0.8**
between terminator separation and next-cycle amplitude. If a sunspot-derivable
proxy retains even part of that relationship, it is worth more than any amount
of additional curve history — and it requires none of the magnetic observations
the original formulation depends on, which is what makes it usable across
twenty-odd cycles rather than the handful that magnetograms cover.

A related classical claim is bundled here: the **Waldmeier effect**, that cycles
which rise faster rise higher, which makes the slope of activity coming out of
minimum a candidate precursor alongside the level.

**H3 — In a scarce-data regime, a strong prior beats a flexible model.**
Estimating a single scalar and rendering the cycle through a fixed parametric
shape is learnable from ~20 examples. Estimating 132 free values is not. The
predicted failure mode is specific and testable: a large free-form network
should regress toward the mean cycle of whatever era it trained on, and should
therefore **systematically under-predict strong cycles** — which is precisely
the direction of the 2020 Cycle 25 miss.

**H4 — Accuracy and calibration must be claimed separately.** Two methods can
tie on error while one assigns 80% intervals that contain the truth 80% of the
time and the other assigns intervals that contain it 37% of the time. Both of
those figures are observed in this work, on methods whose point errors are
statistically indistinguishable. Interval coverage is therefore reported as a
first-class result, not as a diagnostic. The open question this raises is
whether a forecaster at this sample size can be *intrinsically* calibrated, or
whether honest intervals can only ever be obtained by widening a bad one after
the fact.

## 6. Data curation and what it revealed

Several of the most consequential findings in this work are about the data
rather than the modeling.

**Missing-day sentinels contaminate monthly means.** Both SILSO and the sunspot
area database encode a missing day as `-1.0` rather than as an empty field.
Averaged naively into a monthly mean, these sentinels drag early-record months
toward — and sometimes below — zero, producing physically impossible negative
activity in exactly the era where observations are sparsest. The correction has
to happen at **native daily resolution, before any aggregation**; masking after
the fact cannot recover the contaminated means. This has a diagnostic
consequence worth recording: a small constant offset that had previously been
applied during preprocessing, and that looked like a modeling choice, was in
fact a compensation for this bug. Data artifacts of this kind tend to be
absorbed into tuned constants and thereby made invisible.

The most telling evidence is in an old dataset profile generated before the
fix. It reports the minimum sunspot number as **−1.00** and, on the very next
line, **"Missing: 0 (0.0%)"**. The profiling had faithfully described a series
in which every gap had been silently converted into a physically impossible
measurement. Nothing downstream could have flagged this, because by every
mechanical check the data were complete.

**The record is not homogeneous across its length.** Early decades rest on a
handful of observers; the daily table carries both the number of contributing
observations and a per-day standard deviation, and both make the heterogeneity
explicit. Mean activity differs substantially by era — roughly 73 across
1818–1956, 106 across 1957–1999, and 68 across 2000–2025 — which means a model
trained on one stretch and scored on another is partly being scored on a shift
in the baseline rather than on forecasting skill. This is also the mechanism
behind the predicted failure in H3: "regress toward the mean cycle" means the
mean cycle *of the training era*, and that quantity moves.

**Scale revisions are not backward-compatible.** The series used is the version
2.0 recalibration, whose values run roughly **1.67× higher** than the version 1
numbers on which much of the classical literature — including the parametric
cycle-shape work — was built. Any amplitude quoted from older sources has to be
rescaled before it can be compared to, or combined with, anything computed here.
This is a live correctness issue, not a historical footnote.

**Auxiliary series cover a minority of the record.** Geomagnetic indices begin
in 1932 and F10.7 in 1947, against a sunspot record starting in 1749. Any
auxiliary channel is therefore absent for most of the available cycles. That
absence is information — a month before 1932 is not a month of zero geomagnetic
activity — and must be represented as *unavailable* rather than imputed to a
value that the model will read as a measurement.

**Derived features are only causal if constructed causally.** This is the
subtlest curation issue. A cycle minimum is easy to locate in hindsight, because
the standard smoothing that reveals it is centered and therefore looks forward.
But a minimum used as a *model input* must be identifiable from trailing
information only, which means it can be confirmed only after enough subsequent
months have passed without a new low — on the order of **18 months**. A minimum
occurring in month *m* does not become usable until month *m + 18*. Hindsight
detection remains entirely legitimate for a different purpose — defining
evaluation folds, choosing backtest origins, drawing plots — and the curation
has to hold those two notions of "where the minimum is" apart rather than
collapsing them into one convenience function.

**Target and error are defined on different series.** Peak amplitude and timing
follow community convention and are defined on the 13-month smoothed series.
Error is reported on raw monthly values, in native sunspot-number units. These
are not interchangeable, and a figure quoted without saying which series it
refers to is ambiguous by roughly the amount of the monthly scatter.

**Splits must be cycle-shaped.** Because the prediction unit is the cycle,
held-out data has to consist of whole cycles whose target windows are entirely
disjoint from any training target. Every fitted quantity — including
normalization statistics, which are easy to overlook — must be derived from the
pre-origin era only. The strictness here is not fastidiousness; it is the
difference between the RMSE of 2.93 that started this re-examination and the
honest figure.

## 7. Scope

**In scope:** cycle-scale forecasts of sunspot activity — amplitude, timing, and
the monthly envelope — issued from a minimum, with intervals whose stated
coverage is empirically verified.

**Out of scope:** short-horizon space weather. Flare prediction, coronal mass
ejection arrival, and geomagnetic storm nowcasting are different problems on
different timescales with different data requirements. Active-region-level
forecasting is likewise excluded, as is anything requiring magnetograms,
helioseismic inversions, or polar field measurements, none of which extend far
enough back to add cycles to a sample of twenty.

**A known gap in the framing.** Comparison is currently made against internal
reference points — climatology, persistence, a precursor regression, and the
2020 paper's own Cycle 25 prediction. It is *not* made against the established
external forecasts a reader would reasonably expect: the NOAA/NASA/ISES Solar
Cycle 25 Prediction Panel consensus, the McNish–Lincoln method, or the
polar-field precursor forecasts of the Schatten and Svalgaard lineage. Since the
central claim of this work is about honest comparison, the absence of the
field's standard comparators is the most obvious thing to fix.

## References

- Benson, B., Pan, W. D., Prasad, A., Gary, G. A., & Hu, Q. (2020). *Forecasting
  Solar Cycle 25 Using Deep Neural Networks.* Solar Physics, 295:65. — the study
  this work re-examines.
- Hathaway, D. H., Wilson, R. M., & Reichmann, E. J. (1994). *The shape of the
  sunspot cycle.* — the parametric cycle shape underlying H3.
- McIntosh, S. W., et al. (2023). *Deciphering solar magnetic activity: the
  solar cycle clock.* — terminator separation as an amplitude precursor,
  underlying H2.
- SILSO, World Data Center for the production, preservation and dissemination of
  the international sunspot number, Royal Observatory of Belgium. Sunspot Number
  version 2.0.
