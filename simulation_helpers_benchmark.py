import numpy as np
from utils import *
from graphSVD import *
import anndata as ad
import GraphPCA as sg
import pandas as pd
import gplsi
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri

def simulation(k=4, 
               iteration=10, 
               n=1000,
               m=50, 
               r=0.05, 
               p=1000, 
               n_clusters=25, 
               phi=0.1, 
               lamb_start=0.001,
               step_size=1.2,
               grid_len=35, 
               eps=1e-05, 
               noise_level=0.01,
               sparsity=False,
               alpha = 1):
    
    spca_acc = []
    spca_U_l2 = []
    spca_V_l2 = []
    spca_U_l1 = []
    spca_V_l1 = []

    bayes_acc = []

    spatial_acc = []
    spatial_U_l2 = []
    spatial_U_l1 = []

    pca_acc = []
    pca_U_l2 = []
    pca_V_l2 = []
    pca_U_l1 = []
    pca_V_l1 = []

    gpca_acc = []
    gpca_U_l2 = []
    gpca_V_l2 = []
    gpca_U_l1 = []
    gpca_V_l1 = []

    i = 0
    while i < iteration:
        try:
            if sparsity == False:
                coords_df = generate_graph_kmeans(n, p, k, r, n_clusters)
                U_true = generate_W_strong(coords_df, n, p, k, r) 
                U_true = U_true.T
                L_true = np.diag(np.random.rand(min(U_true.shape)))
                V_true, _ = np.linalg.qr(np.random.randn(p, U_true.shape[1]))
                X = U_true @ L_true @ V_true.T
                X = X + np.random.normal(0, noise_level, X.shape)
                weights, edge_df = generate_weights_edge(coords_df, k, 0.05)
            else:
                X, U_true, L_true, V_true, coords_df, edge_df, weights = generate_data(n, m, p, k, r, n_clusters, alpha, noise_level)

            X = euclidean_proj_simplex(X, s=1)
            X = multinomial_from_rows(X,n=100)
            X = np.asarray(X, dtype=np.float64)
            
            pandas2ri.activate()
            numpy2ri.activate()
            robjects.globalenv['k'] = k

            ### SpatialPCA
            counts = X.T  
            location = coords_df.iloc[:, :2].to_numpy()  

            counts_r = numpy2ri.py2rpy(counts)
            location_r = numpy2ri.py2rpy(location)
            robjects.globalenv['counts_r'] = counts_r
            robjects.globalenv['location_r'] = location_r

            robjects.r('rownames(counts_r) <- 1:nrow(counts_r)')
            robjects.r('colnames(counts_r) <- 1:ncol(counts_r)')
            robjects.r('rownames(location_r) <- 1:nrow(location_r)')

            start_time_3 = time.time()
            if sparsity==False:
                robjects.r('''
                library(SpatialPCA)

                ST <- CreateSpatialPCAObject(counts = counts_r,
                    location = location_r,
                    project = "SpatialPCA",
                    gene.type = "spatial",
                    sparkversion = "spark",
                    min.loctions = 2,
                    min.features = 5
                )

                ST <- SpatialPCA_buildKernel(
                    ST,
                    kerneltype = "gaussian",
                    bandwidthtype = "SJ"
                )

                ST <- SpatialPCA_EstimateLoading(
                    ST,
                    fast = FALSE,
                    SpatialPCnum = k
                )

                ST <- SpatialPCA_SpatialPCs(
                    ST,
                    fast = FALSE
                )
                ''')
                SpatialPC = robjects.r('ST@SpatialPCs')
                spatial_acc.append(group_and_compare_spectral(SpatialPC.T, coords_df))
                spatial_U_l2.append(calculate_l2(U_true, SpatialPC.T, k))
                spatial_U_l1.append(calculate_l1(U_true, SpatialPC.T, k))
            else:
                spatial_acc.append(0)
                spatial_U_l2.append(0)
                spatial_U_l1.append(0)

            ### SpatialSVD and Regular PCA
                
            numpy2ri.activate()
            pandas2ri.activate()
            seurat = rpackages.importr('Seurat')
            r_X_T = numpy2ri.py2rpy(X.T)
            robjects.globalenv['counts_matrix'] = r_X_T
            robjects.r('''
                library(Seurat)
                seurat_obj <- CreateSeuratObject(counts = as.matrix(counts_matrix))
                seurat_obj <- SCTransform(seurat_obj, verbose = FALSE)
                normalized_matrix <- as.data.frame(GetAssayData(seurat_obj, layer = "data", assay = "SCT"))
                    ''')

            normalized_df = pandas2ri.rpy2py(robjects.globalenv['normalized_matrix'])
            X_norm = normalized_df.to_numpy().T
            X_norm = X_norm - np.mean(X_norm, axis=0)

            model_spca = gplsi.GpLSI_(
                    lamb_start=lamb_start,
                    step_size=step_size,
                    grid_len=grid_len,
                    initialize=True,
                    sparsity=sparsity,
                    fast_option=True,
                    eps=eps
            )
            model_spca.fit(X_norm, k, edge_df, weights)

            U_hat = model_spca.U
            V_hat = model_spca.V

            spca_acc.append(group_and_compare_spectral(U_hat, coords_df))
            spca_U_l2.append(calculate_l2(U_true, U_hat, k))
            spca_U_l1.append(calculate_l1(U_true, U_hat, k))
            spca_V_l2.append(calculate_l2(V_true, V_hat, k))
            spca_V_l1.append(calculate_l1(V_true, V_hat, k))


            U_pca, _, V_pca = svds(X.astype(float), k=k)
            V_pca = V_pca.T
            pca_acc.append(group_and_compare_spectral(U_pca, coords_df))
            pca_U_l2.append(calculate_l2(U_true, U_pca, k))
            pca_U_l1.append(calculate_l1(U_true, U_pca, k))
            pca_V_l2.append(calculate_l2(V_true, V_pca, k))
            pca_V_l1.append(calculate_l1(V_true, V_pca, k))

            ### BayesSpace
            SingleCellExperiment = rpackages.importr('SingleCellExperiment')
            BayesSpace = rpackages.importr('BayesSpace')

            coords_df_r = coords_df.iloc[:, :2]
            coords_df_r.columns = ['row', 'col']
            location_df_r = pandas2ri.py2rpy(coords_df_r)

            robjects.globalenv['counts_matrix'] = X.T 
            robjects.globalenv['location_df'] = location_df_r

            start_time_2 = time.time()
            robjects.r('''
            rownames(counts_matrix) <- 1:nrow(counts_matrix)
            colnames(counts_matrix) <- 1:ncol(counts_matrix)
            rownames(location_df) <- 1:ncol(counts_matrix)

            sce <- SingleCellExperiment(
            assays = list(counts = counts_matrix),
            colData = location_df
            )

            sce <- spatialPreprocess(sce, platform = "ST", 
                                    n.PCs = k, n.HVGs = 2000, log.normalize = TRUE)

            sce <- spatialCluster(sce, q = k, platform = "ST", d = k,
                                init.method = "mclust", model = "t", gamma = 2,
                                nrep = 10000, burn.in = 100, save.chain = TRUE)
            ''')

            clusters = robjects.r('sce$spatial.cluster')
            bayes_acc.append(get_accuracy(clusters-1, coords_df))


            ### GraphPCA
            adata = ad.AnnData(X=X)
            coords = coords_df[['x', 'y']].to_numpy()  # or whatever your coordinate columns are
            adata.obsm['spatial'] = coords
            Z_gpca, W_gpca = sg.Run_GPCA(
                adata=adata,
                location=adata.obsm['spatial'],
                n_components=k,     
                method="knn",      
                _lambda=0.5,         # tune this hyperparameter
                n_neighbors=k,    
                save_reconstruction=False)
            
            gpca_acc.append(group_and_compare_spectral(Z_gpca, coords_df))
            gpca_U_l2.append(calculate_l2(U_true, Z_gpca, k))
            gpca_U_l1.append(calculate_l1(U_true, Z_gpca, k))
            gpca_V_l2.append(calculate_l2(V_true, W_gpca, k))
            gpca_V_l1.append(calculate_l1(V_true, W_gpca, k))


            i+=1

        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            print("Retrying the current iteration...")
            continue  # Restart the same iteration


    return {
        "spca_acc": np.mean(spca_acc),
        "spca_U_l2": np.mean(spca_U_l2),
        "spca_V_l2": np.mean(spca_V_l2),
        "bayes_acc": np.mean(bayes_acc),
        "spatial_acc": np.mean(spatial_acc),
        "spatial_U_l2": np.mean(spatial_U_l2),
        "pca_acc": np.mean(pca_acc),
        "pca_U_l2": np.mean(pca_U_l2),
        "pca_V_l2": np.mean(pca_V_l2),
        "gpca_acc": np.mean(gpca_acc),
        "gpca_U_l2": np.mean(gpca_U_l2),
        "gpca_V_l2": np.mean(gpca_V_l2)}
