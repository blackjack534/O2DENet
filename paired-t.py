import numpy as np
from scipy import stats

# example
baseline = np.array([0.401, 0.398, 0.405, 0.399, 0.402])
method = np.array([0.465, 0.470, 0.462, 0.468, 0.471])

# evaluation mean and std
print("Baseline mean ± std:", baseline.mean(), baseline.std())
print("Method mean ± std:", method.mean(), method.std())

# paired t
t_stat, p_value = stats.ttest_rel(method, baseline)

print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Statistically significant (p < 0.05)")
else:
    print("Not statistically significant")
