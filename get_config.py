param_sets = [
    {
        "nsim": [5],
        "m": [100, 200, 500],
        "n": [1000],
        "K": [3],
        "p": [1000, 1500, 3000, 5000],
        "n_clusters": [30],
        "alpha": [1]
    },
    {
        "nsim": [5],
        "m": [100],
        "n": [1000],
        "K": [3, 5, 10],
        "p": [1000, 1500, 3000, 5000],
        "n_clusters": [30],
        "alpha": [1]
    },
    {
        "nsim": [5],
        "m": [100],
        "n": [1000],
        "K": [3],
        "p": [1000, 1500, 3000, 5000],
        "n_clusters": [30, 50, 100],
        "alpha": [1]
    },
    {
        "nsim": [5],
        "m": [100],
        "n": [1000],
        "K": [3],
        "p": [1000, 1500, 3000, 5000],
        "n_clusters": [30],
        "alpha": [1, 3, 5]
    }
]


from sklearn.model_selection import ParameterGrid
import pandas as pd

res = pd.DataFrame()
for exper in param_sets:
    grid = ParameterGrid(exper)
    for params in grid:
        res = pd.concat([res, pd.DataFrame(params, index=[0])], axis=0)
# remove duplicates
res = res.drop_duplicates()
res["task_id"] = range(1, len(res) + 1)
res = res[["task_id", "nsim", "m", "n", "K", "p", "n_clusters","alpha"]]
res.to_csv("config.txt", index=False, sep=" ")
