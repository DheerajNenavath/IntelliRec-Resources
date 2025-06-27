
# Simple Hybrid Recommender combining CF and CBF scores
import numpy as np

# Assume scores from collaborative and content-based models
cf_scores = np.array([0.2, 0.6, 0.9, 0.4])
cbf_scores = np.array([0.3, 0.5, 0.8, 0.6])

# Hybrid score (weighted average)
hybrid_scores = 0.5 * cf_scores + 0.5 * cbf_scores

# Recommended item indices sorted by hybrid score
recommended_indices = np.argsort(hybrid_scores)[::-1]
print("Top recommendations (by index):", recommended_indices)
