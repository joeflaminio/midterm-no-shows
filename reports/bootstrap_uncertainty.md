# Saved-prediction uncertainty check

Source: reports/final_predictions.csv at main commit 89da66b2f0c1327f23ee31717bf54efbbe3b1916.
No model is loaded, fitted, selected, retuned, or rerun.

Method: paired IID row bootstrap, 5,000 samples of 1,000 rows with replacement.
The observed outcome and fixed decision (probability >= 0.45) stay paired.
Seed: 42; RNG: Mulberry32 (implementation below).
Intervals: linearly interpolated 2.5th and 97.5th percentiles of total savings.
These intervals condition on the already-fitted model and synthetic test sample.
They omit training/model-selection uncertainty, population shift, and uncertainty
in the assumption that every call prevents a missed visit. They are not evidence
of real-clinic effectiveness.

Verified totals: 1,000 rows, 262 no-shows, 485 calls.
Costs: model $6,470; call nobody $9,170; call everybody $8,000.
Savings:
- Versus nobody: $2,700; 95% percentile interval $1,946.925 to $3,474.025.
- Versus everybody: $1,530; interval $964.975 to $2,056.

This independently corroborates the earlier approximate README ranges; the
original run's seed was not recorded, so it is not an exact reconstruction of it.

## Reproduce from repository root using Node.js

Save the following as a temporary .cjs file and run it from the repository root.
It reads only the saved CSV and prints JSON; it does not change project outputs.

```javascript
const fs = require("node:fs");
const rows = fs.readFileSync("reports/final_predictions.csv", "utf8")
  .trim().split(/\r?\n/).slice(1).map(s => s.split(",").map(Number));
if (rows.length !== 1000 || rows.some(r => r.length !== 3 ||
    !r.every(Number.isFinite) || ![0, 1].includes(r[1]) ||
    r[2] < 0 || r[2] > 1)) throw Error("Unexpected prediction data");
const savings = rows.map(([, y, p]) => {
  const model = p >= 0.45 ? 8 : 35 * y;
  return [35 * y - model, 8 - model];
});
let seed = 42;
function random() {
  let t = seed += 0x6D2B79F5;
  t = Math.imul(t ^ t >>> 15, t | 1);
  t ^= t + Math.imul(t ^ t >>> 7, t | 61);
  return ((t ^ t >>> 14) >>> 0) / 4294967296;
}
const none = [], all = [];
for (let b = 0; b < 5000; b++) {
  let a = 0, c = 0;
  for (let j = 0; j < rows.length; j++) {
    const pair = savings[Math.floor(random() * rows.length)];
    a += pair[0]; c += pair[1];
  }
  none.push(a); all.push(c);
}
function interval(values) {
  values.sort((a, b) => a - b);
  return [0.025, 0.975].map(q => {
    const i = (values.length - 1) * q, j = Math.floor(i);
    return values[j] + (values[Math.min(j + 1, values.length - 1)] -
      values[j]) * (i - j);
  });
}
console.log(JSON.stringify({seed: 42, samples: 5000,
  vs_nobody: interval(none), vs_everybody: interval(all)}, null, 2));
```
