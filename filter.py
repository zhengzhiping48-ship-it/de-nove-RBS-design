import numpy as np
import pandas as pd


df_raw = pd.read_excel("filter_data.xlsx")

generated_seqs = df_raw["generated_seqs"].astype(str).values
pred_strengths = df_raw["pred_strengths"].astype(float).values

generated_seqs = np.array(generated_seqs)
pred_strengths = np.array(pred_strengths)


s_min = pred_strengths.min()
s_max = pred_strengths.max()

# Avoid division by zero (extreme case)
if s_max == s_min:
    norm_strengths = np.ones_like(pred_strengths) * 50.0
else:
    norm_strengths = 1 + 99 * (pred_strengths - s_min) / (s_max - s_min)

# =======================================================
# Select Top-K sequences closest to the target strength
# Distance = |pred_strength - target|
# =======================================================
target_strength = 5.0  # target strength (original scale)
K = 5                 # number of sequences to keep

dist = np.abs(pred_strengths - target_strength)
topk_idx = np.argsort(dist)[:K]

topk_seqs = generated_seqs[topk_idx]
topk_strengths_raw = pred_strengths[topk_idx]
topk_strengths_norm = norm_strengths[topk_idx]
topk_dist = dist[topk_idx]

df_out = pd.DataFrame({
    "sequence": topk_seqs,
    "pred_strength_raw": topk_strengths_raw,
    "pred_strength_norm_1_100": topk_strengths_norm,
    "distance_to_target": topk_dist
})

df_out.to_csv("topk_rbs.csv", index=False)

print(df_out.head())
