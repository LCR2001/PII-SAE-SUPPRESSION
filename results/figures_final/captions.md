# Final Paper Figure Captions

All figures are saved as PNG (300 dpi), PDF, and SVG in this directory.
Selectivity = |hero mean| / max(|random-vector mean|, 0.01). 95% confidence
intervals are computed with the Student-t approximation 1.96·SE (n=50 unless
noted). Sign convention: α=−2 for Gemma task-SAE and pre-trained Gemma SAE,
α=+2 for pre-trained Llama L25 (sign-reversed, V3 5a-extended convention).

---

## Figure 1 — Layer Selectivity (Gemma, task-SAE)

**File:** `fig1_layer_selectivity.{png,pdf,svg}`

Cross-template selectivity of the top-1 task-SAE feature at three Gemma-2-9B
layers: **L9** (PCA-peak layer, green, feature 20), **L20** (yellow, feature
670), and **L31** (red, feature 2813). Each bar shows |Δ|hero / |Δ|random-vector
on the natural-name PII span at α=−2, evaluated over n=50 validation pairs per
template (T0–T8). **(a)** selectivity computed on Δlogprob; **(b)** on
Δb_rank (sum log₂(rank) bits over the PII span). The horizontal dashed line
marks the 5× working threshold used throughout the paper, the dotted line
marks parity. L9 exceeds 5× on every template for both metrics, whereas L20
and L31 hover at parity, indicating that the PCA-peak layer concentrates the
PII-feature direction while later layers do not. This single panel justifies
PCA-peak layer selection rather than uniform layer choice.

---

## Figure 2 — ID Mismatch Pattern (Llama-Med L25)

**File:** `fig2_id_mismatch.{png,pdf,svg}`

Pre-trained Llama3-Med42-8B SAE feature 22072 (L25, α=+2) on the patient-id
PII span across nine templates, n=50/template. **(a)** mean Δlogprob with
95% CI; **(b)** mean Δb_rank (bits) with 95% CI. Red bars mark *mismatch*
templates — those where Δlogprob's CI overlaps zero **but** Δb_rank is
significantly below zero. Three templates (T1, T4, T5) satisfy this strict
criterion: the gold token's mean log-probability is preserved while its rank
in the model's full output distribution degrades, an effect-on-tail-mass that
a mean-only logprob metric misses entirely. T3 and T7 show the conventional
"primary suppression" pattern (both Δlp and Δb_rank significantly negative);
T8 shows an opposite, anti-suppression Δlp paired with degraded ranks. The
yellow text box summarises the mismatch count and interpretation. This figure
motivates the multi-metric framework: a reviewer relying only on Δlp would
declare these conditions null.

---

## Figure 3 — Magnitude Artifact Test (Task-SAE vs Pre-trained SAE)

**File:** `fig3_artifact_comparison.{png,pdf,svg}`

Side-by-side comparison of two top-1 candidates on Gemma-2-9B × full_name
PII, n=50/template, α=−2. Pre-trained Gemma-Scope SAE L10 feature 2759 (gray)
vs task-specific SAE L9 feature 20 (blue). **(a)** hero |Δlogprob| (nat) per
template: the two features achieve comparable raw magnitudes on most
templates (T5 spike from the Korean-variant template format affects both).
**(b)** selectivity (log scale) at matched magnitude: the pre-trained feature
sits near 1× — its hero effect is barely distinguishable from a random
direction of the same norm — whereas the task-SAE feature sits at 7–20×,
well above the 5× threshold. T8 reaches a ~20× ratio (annotation). The same
hero magnitude does not imply the same feature specificity; without random-
direction controls a pre-trained SAE feature can pass a Δlp-only screen
while in fact behaving as a magnitude artifact.

---

## Figure 4 — External Utility on PubMedQA

**File:** `fig4_pubmedqa_utility.{png,pdf,svg}`

Hero PII interventions evaluated on the PubMedQA labeled test split (n=500
yes/no/maybe questions). Each question is scored by single-token logprob over
{yes, no, maybe} at the answer position; the argmax is taken as the
prediction. **(a)** baseline (gray) vs hook-on (colored) accuracy per
condition; Δacc annotated above each pair, 95% Wald confidence interval. All
four conditions preserve accuracy within ±1.4pp despite causing substantial
PII recovery degradation (paper Fig 1–3). **(b)** off-target intervention
magnitude — bar = mean |Δlp| over the three answer tokens (bar height shown
inside / above bar), line+marker = prediction flip rate (right axis, red).
Only the Gemma task-SAE L9 f=20 condition perturbs the answer-position
distribution meaningfully (3.6 nat, 28% flip rate), and even there accuracy
loss is just 1.4pp; the Llama hero features barely activate on PubMedQA
prompts (|Δlp|<0.03), additional evidence of feature specificity. Note
baselines reflect zero-shot PubMedQA difficulty — Gemma defaults toward
"yes" (~56% acc ≈ majority-class), Llama-Med42 reaches ~71%; these are
unrelated to the intervention test and serve only as utility-preservation
references.

---

## Figure A1 — α Sensitivity (Reviewer Defense Exp 1)

**File:** `figA1_alpha_sensitivity.{png,pdf,svg}`

Four-panel 2×2 grid demonstrating that the paper's hero findings are stable
across the α magnitudes tested, ruling out an α-tuning artifact. Each panel
plots mean Δlogprob (nat) with 95% CI vs |α|, hero feature (solid) and a
norm-matched random-direction control (dashed gray) on the same residual
stream, n=50 validation pairs per (α, template). **(a)** pre-trained Gemma
SAE L10 f=2759 × name × T1, α∈{−1,−2,−3}; **(b)** task-SAE Gemma L9 f=20
× name × T1, same α range; **(c)** pre-trained Llama L25 f=22072 × name × T0,
α∈{+1,+2,+3} (sign-reversed convention); **(d)** the same Llama feature on
id × T3 — the mismatch case — at α∈{+1,+2,+3}. In all panels the hero–
random gap widens monotonically with |α| while the random control stays near
zero, confirming that feature specificity is robust to α and not a magnitude
coincidence.

---

## Figure A2 — PCA-based Layer Selection

**File:** `figA2_pca_silhouette.{png,pdf,svg}`

4-way silhouette score (full_name / patient_id / phone / email
embeddings) per residual-stream layer for Gemma-2-9B (left) and
Llama3-Med42-8B (right), comparing pre-trained base (gray) and our LoRA-
merged fine-tuned model (red). Selected intervention layers — Gemma L10 and
Llama L25, both fine-tuned-peak — are highlighted; V3-paper-baseline layers
(Gemma L40, Llama L30) are shown for reference. Fine-tuning reshapes the
PII-cluster geometry, shifting peak separation to earlier layers; this
silhouette-peak criterion is what motivates the new paper's task-SAE
layer-discovery procedure rather than reusing V3's downstream layers.
