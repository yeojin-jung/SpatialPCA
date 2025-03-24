from ast import main
import time
import numpy as np
from numpy.linalg import norm

from generate_spca_data import *
from utils import *
from graphSVD import *
import gplsi

from collections import defaultdict

def run_simul(
    nsim = 5,
    n = 1000,
    p = 1000,
    m = 50,
    k = 3,
    r = 0.05,
    n_clusters = 30,
    phi=0.1,
    lamb_start=0.001,
    step_size=1.2,
    grid_len=35,
    eps=1e-05,
    start_seed=None,
):
    results = defaultdict(list)
    models = {'spca': [], 'init-only': [], 'pca': []}

    
    for trial in range(nsim):
        os.system(f"echo Running trial {trial}...")
        if not (start_seed is None):
            np.random.seed(start_seed + trial)
            X, U_true, Lambda_true, V_true, coords_df, edge_df, weights = generate_data(n, m, p, k, r, n_clusters)

        # SPCA
        start_time = time.time()
        model_spca = gplsi.GpLSI_(
                lamb_start=lamb_start,
                step_size=step_size,
                grid_len=grid_len,
                initialize=True,
                sparsity=True,
                fast_option=True,
                eps=eps
            )
        model_spca.fit(X, k, edge_df, weights)
        time_spca= time.time() - start_time
        print(f"CV Lambda is {model_spca.lambd}")

        # SPCA without sparsity

        # PCA
        U_pca, L_pca, V_pca = svds(X, k=k)
        V_pca  = V_pca.T
        L_pca = np.diag(L_pca)

        # Record errors
        # Calculate errors for model_spca
        err_U_spca_l1 = calculate_l1(U_true, model_spca.U, k)
        err_U_spca_l2 = calculate_l2(U_true, model_spca.U, k)

        err_V_spca_l0 = calculate_l0(V_true, model_spca.V, m, k)
        err_V_spca_l1 = calculate_l1(V_true, model_spca.V, k)
        err_V_spca_l2 = calculate_l2(V_true, model_spca.V, k)

        # Calculate errors for init
        err_U_init_l1 = calculate_l1(U_true, model_spca.U_init, k)
        err_U_init_l2 = calculate_l2(U_true, model_spca.U_init, k)

        err_V_init_l0 = calculate_l0(V_true, model_spca.V_init, m, k)
        err_V_init_l1 = calculate_l1(V_true, model_spca.V_init, k)
        err_V_init_l2 = calculate_l2(V_true, model_spca.V_init, k)

        # Calculate errors for pca
        err_U_pca_l1 = calculate_l1(U_true, U_pca, k)
        err_U_pca_l2 = calculate_l2(U_true, U_pca, k)

        err_V_pca_l0 = calculate_l0(V_true, V_pca, m, k)
        err_V_pca_l1 = calculate_l1(V_true, V_pca, k)
        err_V_pca_l2 = calculate_l2(V_true, V_pca, k)

        # Append results
        results["trial"].append(trial)
        results["n"].append(n)
        results["p"].append(p)
        results["m"].append(m)
        results["k"].append(k)
        results["n_clusters"].append(n_clusters)

        results['spca_err_U_l1'].append(err_U_spca_l1)
        results['spca_err_U_l2'].append(err_U_spca_l2)

        results['spca_err_V_l0'].append(err_V_spca_l0)
        results['spca_err_V_l1'].append(err_V_spca_l1)
        results['spca_err_V_l2'].append(err_V_spca_l2)

        results['init_err_U_l1'].append(err_U_init_l1)
        results['init_err_U_l2'].append(err_U_init_l2)

        results['init_err_V_l0'].append(err_V_init_l0)
        results['init_err_V_l1'].append(err_V_init_l1)
        results['init_err_V_l2'].append(err_V_init_l2)

        results['pca_err_U_l1'].append(err_U_pca_l1)
        results['pca_err_U_l2'].append(err_U_pca_l2)

        results['pca_err_V_l0'].append(err_V_pca_l0)
        results['pca_err_V_l1'].append(err_V_pca_l1)
        results['pca_err_V_l2'].append(err_V_pca_l2)

        results["spca_time"].append(time_spca)

        models['spca'].append(model_spca)

    results = pd.DataFrame(results)
    return results

if __name__ == "__main__":
    p_list = [500, 1000, 1500, 3000, 5000]
    nclusters_list = [30, 50, 100]
    #nclusters_list = [30, 50, 100, 150, 300]
    n = 1000
    m = 50
    k = 3
    r = 0.05

    results_dir = os.path.join(os.getcwd(), "output")
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except:
            pass

    for p in p_list:
        for n_clusters in nclusters_list:
            msg = "Running experiment with n={}, p={}, m={}, k={}, n_clusters={}".format(n, p, m, k, n_clusters)
            print(msg)
            results = run_simul(
                nsim = 5,
                n = n,
                p = p,
                m = m,
                k = k,
                r = r,
                n_clusters = n_clusters
            )
            results_csv_loc = os.path.join(results_dir, f"results_n={n}_p={p}_m={m}_k={k}_n_clusters={n_clusters}.csv")
            results.to_csv(
                results_csv_loc,
                mode="a",
                header=not os.path.exists(results_csv_loc),
                index=False,
            )