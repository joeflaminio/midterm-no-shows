### Training-only model comparison and tuning

Using the corrected supplied generator and Joe’s shared pipeline, I compared a majority-class baseline, balanced logistic regression, decision tree and random forest. I used only the 3,000-row training partition, with five-fold stratified cross-validation and seed 11. Preprocessing stayed inside the pipeline. F1 was selected before comparing results.

Mean F1 ± fold standard deviation was:
- Majority baseline: **0.0000 ± 0.0000**
- Balanced logistic regression: **0.5150 ± 0.0151**
- Decision tree: **0.3637 ± 0.0153**
- Random forest: **0.3017 ± 0.0401**

Balanced logistic regression led on F1. I tuned two parameters across six combinations: `C` = 0.1, 1 or 10, and solver = `liblinear` or `lbfgs`. The best combination, `C=0.1` and `lbfgs`, achieved **0.5168 ± 0.0133** F1. The gain of **0.0018** is small relative to fold variation and does not establish a meaningful improvement.

I saved 3,000 training-only out-of-fold probabilities for later threshold selection. No cost threshold has been chosen, and the final holdout has not been evaluated. These are development results, not final performance claims.
