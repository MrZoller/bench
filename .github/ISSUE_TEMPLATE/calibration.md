---
name: Calibration measurement
about: A predicted-versus-measured pair from your own hardware
title: 'calibration: <model> on <machine>'
labels: calibration
---

<!--
The "Check these numbers" panel on the site fills this in for you — paste your
llama-bench output there and use the submission link. Filling it by hand is fine
too; the fields below are what make a measurement usable.
-->

**Scenario:** <!-- REQUIRED. The share link from the site. -->

<!--
This field is not optional and it is not paperwork. llama-bench names the model
file and the backend, but not the host reliably — so the scenario link is the
only thing tying a measurement to a device row, a quantization and a context. A
measurement that cannot name its scenario cannot be compared with anything, and
will be closed rather than absorbed.
-->

**Machine:**
**Model:**
**llama.cpp build:** <!-- The commit. `-o json` captures it; markdown does not. -->

| measure | tokens | depth | predicted t/s | measured t/s | error |
| ------- | -----: | ----: | ------------: | -----------: | ----: |
|         |        |       |               |              |       |

**Anything unusual about the run?**

<!--
Thermals, a laptop on battery, a driver version, another process on the GPU.
Worth saying: the record is meant to show the distribution rather than the
flattering pairs, and an outlier with a stated cause is more useful than a
tidy number with none.
-->

---

<!--
What happens to this, so nobody is surprised:

Measurements accumulate as evidence. The constants are NOT retuned to chase
incoming submissions one at a time — that is how the next masked error gets
fitted into them silently, and docs/ROADMAP.md records the rule at length. What
submissions in bulk unlock is `bandwidthEfficiency` x `CLASS_BANDWIDTH_UTILIZATION`
becoming identifiable at all, which needs more than one CPU-capable data point.

Misses are published with the same weight as hits.
-->
