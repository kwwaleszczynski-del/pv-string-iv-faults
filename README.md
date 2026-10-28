# PV string fault diagnosis from inverter MPPT voltage and current

[![tests](https://github.com/kwwaleszczynski-del/pv-string-iv-faults/actions/workflows/ci.yml/badge.svg)](https://github.com/kwwaleszczynski-del/pv-string-iv-faults/actions/workflows/ci.yml)

When a PV string loses 3 % to 6 % of its power, the monthly energy waterfall in
[pv-budget-bridge](https://github.com/kwwaleszczynski-del/pv-budget-bridge) puts that deficit into the unexplained bar,
and an availability monitor such as [pv-outage-classifier](https://github.com/kwwaleszczynski-del/pv-outage-classifier)
sees the MPPT channel still running. Taking an I-V curve tracer to the field separates the causes, but string inverters
already log 15-minute operating voltage $`V_{\text{mppt},k}(t)`$ and current $`I_{\text{mppt},k}(t)`$ on every MPPT input.

This repository separates five physical string faults from 15-minute $`(V_{\text{mppt}}, I_{\text{mppt}}, G_{\text{poa}}, T_{\text{amb}})`$
telemetry alone, and checks each estimator against faults of known physical size injected into a 72-sub-string
single-diode and reverse-bias breakdown model (Bishop 1988):

- **shorted bypass diodes** ($`n_{\text{bd}}`$ failed sub-strings): a constant voltage step
  $`\Delta V \approx 14.3\text{ V} \cdot n_{\text{bd}}`$ independent of string current ($`\partial \Delta V / \partial I \approx 0`$)
  with unchanged current ($`i_k \approx 1.000`$);
- **series-resistance degradation** ($`\Delta R_s`$ from corroded MC4 connectors or solder joints): an ohmic voltage drop
  $`\Delta V(t) = \Delta R_s \cdot I_{\text{mppt},k}(t)`$ that vanishes at low morning/evening irradiance and peaks at
  midday high current;
- **potential-induced degradation (PID / shunt conductance $`G_{\text{sh}}`$)**: a voltage ratio $`v_k`$ that droops most
  strongly at low irradiance ($`G_{\text{poa}} < 300\text{ W/m}^2`$) as shunt leakage across the negative-end modules
  steals a larger share of the photocurrent;
- **morning or evening obstacle shading**: bimodal switching between the bypassed low-voltage global MPPT peak and the
  unbypassed high-voltage peak inside a fixed solar azimuth and elevation sector;
- **uniform soiling**: a linear current deficit ($`i_k < 1`$) across all irradiance bins with untouched
  temperature-corrected voltage ($`v_k \approx 1.000`$) and rain-wash step recoveries.

**Interactive page:** https://kwwaleszczynski-del.github.io/pv-string-iv-faults/

> Working material. SCADA MPPT diagnostics isolate the affected MPPT input and classify the electrical signature so a
> field crew knows whether to bring replacement MC4 connectors, a thermal camera, an electroluminescence (EL) rig or a
> cleaning crew. Confirm every fault with an on-site measurement and isolation test before opening energized DC strings.

## Results in one table

12-MPPT commercial string inverter (`24 x 550 Wp` half-cut mono-PERC modules per string, `3` sub-strings per module =
`72` sub-strings per string, $`V_{\text{mp,STC}} = 1,003.2\text{ V}`$, $`I_{\text{mp,STC}} = 13.16\text{ A}`$,
$`158.4\text{ kWp}`$ total DC) over one year of 15-minute operation (`35,040` quarter-hours, `results/mppt_channel_diagnostics.csv`):

| Channel | Injected fault | Diagnosed class | True $`n_{\text{bd}}`$ | Est. $`\hat{n}_{\text{bd}}`$ | $`\Delta V_0`$ intercept [V] | True $`\Delta R_s`$ [$`\Omega`$] | Est. $`\widehat{\Delta R_s}`$ [$`\Omega`$] | Low-$`G`$ $`V`$ drop [%] | Median $`i_k`$ [-] | Energy [MWh] | True loss [MWh] | Est. loss [MWh] |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MPPT 1 | Healthy | Healthy | 0 | 0 | -0.17 | 0.00 | 0.00 | -0.03 | 1.001 | 21.84 | 0.00 | 0.00 |
| MPPT 2 | 2 shorted bypass diodes (May) | Shorted Bypass Diode | 2 | 2 | 28.63 | 0.00 | 0.00 | 2.87 | 1.001 | 21.39 | 0.45 | 0.43 |
| MPPT 3 | Corroded MC4 (+2.5 $`\Omega`$ ramp, Jul) | High Series Resistance R_s | 0 | 0 | -0.15 | 2.50 | 2.47 | 0.66 | 0.997 | 21.61 | 0.22 | 0.21 |
| MPPT 4 | PID shunt loss ($`R_{\text{sh}} = 35\,\Omega`$, Jun) | PID / Low R_sh | 0 | 0 | 37.41 | 0.00 | 0.00 | 3.23 | 0.992 | 21.48 | 0.35 | 0.34 |
| MPPT 5 | Morning obstacle shading (6 sub-strings) | Partial Shading | 0 | 0 | 3.63 | 0.00 | 0.00 | -0.03 | 1.001 | 21.52 | 0.32 | 0.30 |
| MPPT 6 | Uniform soiling (4.5 %, rain-washed) | Uniform Soiling | 0 | 0 | 0.55 | 0.00 | 0.00 | 0.03 | 0.956 | 20.89 | 0.95 | 0.93 |
| MPPT 7 | 1 shorted bypass diode (Apr) | Shorted Bypass Diode | 1 | 1 | 14.23 | 0.00 | 0.00 | 1.42 | 1.001 | 21.56 | 0.28 | 0.26 |
| MPPT 8 | 4 shorted bypass diodes (Jun) | Shorted Bypass Diode | 4 | 4 | 57.41 | 0.00 | 0.00 | 5.76 | 1.001 | 21.14 | 0.70 | 0.68 |
| MPPT 9 | Solder joint (+1.5 $`\Omega`$ $`R_s`$, Apr) | High Series Resistance R_s | 0 | 0 | -0.20 | 1.50 | 1.48 | 0.38 | 0.999 | 21.56 | 0.28 | 0.26 |
| MPPT 10 | Severely corroded MC4 (+3.2 $`\Omega`$ $`R_s`$, May) | High Series Resistance R_s | 0 | 0 | -0.15 | 3.20 | 3.17 | 0.86 | 0.996 | 21.29 | 0.55 | 0.53 |
| MPPT 11 | Healthy | Healthy | 0 | 0 | -0.14 | 0.00 | 0.00 | -0.03 | 1.001 | 21.84 | 0.00 | 0.00 |
| MPPT 12 | Healthy | Healthy | 0 | 0 | -0.17 | 0.00 | 0.00 | -0.03 | 1.001 | 21.84 | 0.00 | 0.00 |

![String I-V and P-V curves](docs/img/iv_pv_curves.png)

## Verification

`python -m pv_iv_faults` runs the 12-MPPT simulation, applies the diagnostic estimators without access to the fault
labels, and writes `results/verification.json`.

| Fault family | Injected physical truth | Recovered from 15-min $`(V_{\text{mppt}}, I_{\text{mppt}}, G_{\text{poa}}, T_{\text{amb}})`$ | Error / false alarms |
|---|---|---|---|
| Shorted bypass diodes ($`n_{\text{bd}} = 1, 2, 4`$) | $`\Delta V = 14.3\text{ V}, 28.6\text{ V}, 57.2\text{ V}`$ | $`\hat{n}_{\text{bd}} = 1, 2, 4`$ (`100 %` integer recovery); intercepts `14.24 V`, `28.63 V`, `57.41 V` | Max intercept error `0.21 V` (`< 0.4 V`) |
| Series resistance ($`\Delta R_s = +1.5\,\Omega, +2.5\,\Omega, +3.2\,\Omega`$) | $`+1.50\,\Omega`$, $`+2.50\,\Omega`$, $`+3.20\,\Omega`$ | $`\widehat{\Delta R_s} = 1.48\,\Omega, 2.47\,\Omega, 3.17\,\Omega`$ | Relative error `-0.9 %` to `-1.3 %` (`< 2 %`), zero diode confusion |
| PID shunt degradation (4 negative-end modules) | $`R_{\text{sh}}: 400\,\Omega \rightarrow 35\,\Omega`$ | Flagged `14 days` after onset once low-$`G`$ ($`< 300\text{ W/m}^2`$) voltage drop crosses `2.2 %` | Steady low-$`G`$ drop `3.23 %` |
| Uniform string soiling | `4.5 %` dry-spell build-up with rain cleanings | `4.4 %` current loss recovered from $`1 - i_k`$ | `0.0 %` false voltage-fault alarms |
| Morning obstacle shading | 6 sub-strings shaded (`105-155 deg` azimuth, `< 28 deg` elevation) | Bimodal voltage branch flagged inside morning sector; midday $`v_k = 1.000`$ | Separated from permanent diode failure |
| 6-class fault classification (`312` bi-weekly channel windows) | `Healthy`, `Shorted Bypass Diode`, `High Series Resistance R_s`, `PID / Low R_sh`, `Partial Shading`, `Uniform Soiling` | Macro-F1 = **`0.968`** | Only transition windows during initial fault ramps miss |

![Diagnostic separation plane](docs/img/dv_vs_current_signature.png)

Why $`\widehat{\Delta R_s}`$ reads $`1.48\,\Omega`$ on a $`1.50\,\Omega`$ fault and $`3.17\,\Omega`$ on a $`3.20\,\Omega`$ fault:
when extra series resistance $`\Delta R_s`$ is added to a string, the inverter MPPT maximizes
$`P(I) = (V_{\text{string},0}(I) - I \Delta R_s) I`$. At the new maximum power point, $`I_{\text{mppt}}`$ shifts slightly
left along the single-diode knee where the internal cell voltage $`V_{\text{string},0}(I)`$ is higher by about
$`0.011 \, I_{\text{mppt}} \Delta R_s`$. The terminal voltage deficit against a healthy peer string is therefore
$`\Delta V_k(I) \approx 0.989 \, \Delta R_s I_{\text{mppt}}`$, so ordinary linear regression of $`\Delta V_k`$ on
$`I_{\text{mppt}}`$ underestimates $`\Delta R_s`$ by $`1.1\,\%`$.

![12-month daily MPPT timeline](docs/img/mppt_fleet_timeline.png)

![Parameter recovery and F1](docs/img/verification_recovery.png)

## Method

### 1. Sub-string single-diode and reverse-bias breakdown model (Bishop 1988)

Each string consists of $`N_{\text{mod}} = 24`$ half-cut mono-PERC 550 Wp modules in series, with 3 sub-strings (and 3
Schottky bypass diodes) per module, giving $`N_{\text{sub}} = 72`$ series sub-strings per string. At cell temperature
$`T_c`$ and effective plane-of-array irradiance $`G_m`$, sub-string $`m`$ with junction voltage
$`V_{d,m} = V_{\text{sub},m} + I_{\text{cell},m} R_{s,\text{sub}}`$ follows the 5-parameter single-diode equation with
Bishop's avalanche breakdown multiplier on the shunt term:

```math
I_{\text{cell},m}(V_{d,m}) = I_{\text{ph},m}(G_m, T_c) - I_{0,m}(T_c)\left[\exp\left(\frac{V_{d,m}}{a_m(T_c)}\right) - 1\right] - \frac{V_{d,m}}{R_{\text{sh,sub},m}}\left[1 + a_{\text{bd}}\left(1 - \frac{V_{d,m}}{V_{\text{br}}}\right)^{-m_{\text{bd}}}\right]
```

A parallel bypass diode conducts forward current $`I_{\text{bp}}(V_{\text{sub},m}) = I_{0,\text{bp}}[\exp(-V_{\text{sub},m}/a_{\text{bp}}) - 1]`$
when the sub-string is driven into reverse bias ($`V_{\text{sub},m} \approx -0.55\text{ V}`$ at $`I_{\text{mp}}`$). Because
all 72 sub-strings carry the same string current $`I`$, inverting $`V_{\text{sub},m}(I)`$ on a monotonic junction-voltage
grid and summing across sub-strings gives the exact string curve:

```math
V_{\text{string}}(I) = \sum_{m=1}^{72} V_{\text{sub},m}(I) - I \cdot \Delta R_s
```

### 2. Peer-string and clear-sky reference normalisation

For each MPPT channel $`k \in \{1, \dots, 12\}`$ and 15-minute daylight interval $`t`$:

```math
v_k(t) = \frac{V_{\text{mppt},k}(t)}{V_{\text{ref}}(G_t, T_{c,t})}, \qquad i_k(t) = \frac{I_{\text{mppt},k}(t)}{I_{\text{ref}}(G_t, T_{c,t})}, \qquad \Delta V_k(t) = V_{\text{peer}}(t) - V_{\text{mppt},k}(t)
```

where $`V_{\text{peer}}(t)`$ and $`I_{\text{peer}}(t)`$ are the medians across the healthy peer cluster on the same
inverter. Comparing against peers on the same roof or mounting table cancels common-mode errors in module temperature,
plane-of-array transposition and spectral shift.

### 3. Diagnostic separation plane

Over unshaded daytime intervals ($`G_{\text{poa}} > 250\text{ W/m}^2`$), linear regression of the peer voltage deficit on
channel current gives:

```math
\Delta V_k(t) = \Delta V_{0,k} + \widehat{\Delta R_{s,k}} \, I_{\text{mppt},k}(t) + \varepsilon_t
```

1. **Shorted bypass diode:** $`|\widehat{\Delta R_{s,k}}| < 0.25\,\Omega`$, $`\Delta V_{0,k} > 8.0\text{ V}`$, and
   low-irradiance vs high-irradiance voltage deficit differs by less than $`6\text{ V}`$. The number of shorted
   sub-strings is recovered as:
   ```math
   \hat{n}_{\text{bd},k} = \mathrm{round}\left(\frac{\Delta V_{0,k}}{V_{\text{mp,STC}} / 72 + V_{\text{diode}}}\right) = \mathrm{round}\left(\frac{\Delta V_{0,k}}{14.30\text{ V}}\right)
   ```
2. **High series resistance ($`\Delta R_s`$):** $`\widehat{\Delta R_{s,k}} \ge 0.45\,\Omega`$ and
   $`|\Delta V_{0,k}| < 5.0\text{ V}`$. The voltage drop vanishes at dawn and dusk and scales linearly with current.
3. **PID shunt degradation:** low-irradiance ($`100 \le G_{\text{poa}} \le 300\text{ W/m}^2`$) voltage deficit exceeds
   high-irradiance ($`G_{\text{poa}} \ge 550\text{ W/m}^2`$) voltage deficit by more than $`5\text{ V}`$, and the
   low-irradiance voltage loss $`1 - v_k(G < 300\text{ W/m}^2)`$ exceeds $`2.2\,\%`$.
4. **Partial shading:** 90th-percentile voltage deficit exceeds $`35\text{ V}`$ inside a compact solar azimuth and
   elevation window while unshaded midday intervals sit at $`|\Delta V_k| < 4.5\text{ V}`$.
5. **Uniform soiling:** current loss $`1 - i_k > 1.8\,\%`$ across all irradiance bins while $`|\Delta V_k| < 3.5\text{ V}`$.

## Limitations

- **Parallel strings on one MPPT ($`8 \times 2`$ topology):** when two strings share one MPPT input (`strings_per_mppt = 2`),
  a shorted bypass diode in one string forces the healthy parallel string off its own maximum power point and splits the
  voltage step roughly in half ($`\sim 7\text{--}9\text{ V}`$ instead of $`14.3\text{ V}`$), making single-diode integer
  counting less sharp than on single-string MPPT inputs.
- **Global MPPT sweep interval:** if an inverter does not run periodic global MPPT sweeps during partial shading, it can
  remain trapped on the high-voltage local peak ($`V \approx V_{\text{peer}}`$, reduced $`I`$), which looks like a brief
  current loss rather than a $`-86\text{ V}`$ bypass voltage step.
- **Peer reference requires unfaulted strings:** if all strings on an inverter degrade uniformly (for example, plant-wide
  PID or uniform soiling across every MPPT), $`V_{\text{peer}}`$ moves with the fault and detection must rely on the
  temperature-corrected model reference $`V_{\text{ref}}(G_t, T_{c,t})`$, which carries $`\pm 1.5\,\%`$ cell-temperature
  uncertainty.
- **No module-level localization within a string:** SCADA MPPT voltage tells how many sub-strings or Ohms are lost on
  channel $`k`$, not which physical module position along the 24-module string holds the shorted diode or hot MC4
  connector.

## Reproduce

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m pv_iv_faults
```

`python -m pv_iv_faults` runs the 1-year 15-minute simulation, writes `results/mppt_channel_diagnostics.csv`,
`results/daily_fleet_tracking.csv` and `results/verification.json`, regenerates the four PNG figures in `docs/img/`, and
writes `docs/data.js` for the interactive page. The 21 unit and verification tests run offline in about 6 seconds.

## References

- Bishop, J. W. (1988). Computer simulation of the effects of electrical mismatches in photovoltaic cell
  interconnection circuits. *Solar Cells*, 25(1), 73-89.
- De Soto, W., Klein, S. A., Beckman, W. A. (2006). Improvement and validation of a model for photovoltaic array
  performance. *Solar Energy*, 80(1), 78-88.
- IEC 61724-1, Photovoltaic system performance - Part 1: Monitoring.
- Pingel, S. et al. (2010). Potential Induced Degradation of solar cells and panels. *35th IEEE PVSC*.

MIT licence.
