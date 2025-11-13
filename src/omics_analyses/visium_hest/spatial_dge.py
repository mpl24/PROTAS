import limix
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
from scipy.linalg import block_diag
from scipy.spatial.distance import pdist, squareform
import os
import multiprocessing as mp
from statsmodels.stats.multitest import multipletests
from functools import partial
from numba import jit, prange
import warnings
import glob
from scipy.linalg import eigvals, LinAlgError

os.environ['OMP_NUM_THREADS'] = '3'
os.environ['MKL_NUM_THREADS'] = '3'
os.environ['OPENBLAS_NUM_THREADS'] = '3'

@jit(nopython = True)
def fast_rbf_kernel(coords, length_scale = 10.0):
    n = coords.shape[0]
    K = np.zeros((n, n))
    for i in prange(n):
        for j in range(i, n):
            dist_sq = 0.0
            for k in range(coords.shape[1]):
                diff = coords[i, k] - coords[j, k]
                dist_sq += diff * diff

            val = np.exp(-dist_sq/(2 * length_scale ** 2))
            K[i, j] = val
            K[j, i] = val

    return K


def cache_spatial_kernels(coords_list, length_scales):
    cached_kernels = {}
    sample_kernels_cache = {}

    for i, coords in enumerate(tqdm(coords_list, desc = 'Per-sample kernels')):
        coords_array = np.ascontinguousarray(coords.astype(np.float64))
        sample_kernels_cache[i] = {}
        for scale in length_scales:
            sample_kernels_cache[i][cache] = fast_rbf_kernel(coords_array, scale)
    
    for scale in tqdm(length_scale, desc = 'Create block diagnonal matrices'):
        kernels = [sample_kernels_cache[i][scale] for i in range(len(coords_list))]
        K_spatial = block_diag(*kernels)
        cached_kernels[scale] = K_spatial
    
    return cached_kernels


def optimize_length_scale_null(
    metadata, 
    K_sample,
    cached_kernels,
    sample_gene_data
    ):
    print('Starting legnth scale optimization!')
    sample_ids = metadata['slide_name'].values
    samples_unique = np.unique(sample_ids)

    sample_matrix = np.zeros(
        (len(metadata), len(samples_unique)))
    
    for i, sample_id in enumerate(sample_ids):
        sample_idx = np.where(samples_unique == sample_id)[0][0]
        sample_matrix[i, sample_idx] = 1
    
    M = np.column_stack(
        [np.ones(len(metadata)),
        sample_matrix[:, 1:]])

    best_avg_ll = -np.inf
    best_length = None

    for length_scale, K_spatial in tqdm(
        cached_kernels.items(), desc = 'testing length scales'):
        K_total = K_spatial + K_sample # combine both kernels for both correlation
        K_total += 1e-2 * np.eye(K_total.shape[0])

        avg_ll = 0
        valid_genes = 0

        for gene_data in sample_gene_data:
            try:
                y = gene_data.reshape(-1, 1)
                lmm = limix.qtl.scan(
                    G = np.zeros((len(metadata), 1)) # dummy 
                    Y = y,
                    M = M,
                    K = K_total,
                    verbose = False
                )
                avg_ll += lmm.stats.lml0[0]
                valid_genes += 1
            except:
                continue

        if valid_genes > 0:
            avg_ll /= valid_genes
            if avg_ll > best_avg_ll:
                best_avg_ll = avg_ll
                best_length = length_scale
    
    print(f'Optimal length scale: {best_length}')
    return best_length


def process_single_sample_gene(gene_data):
    gene, y, M, G, K_spatial = gene_data

    try:
        lmm = limix.qtl.scan(
            G = G,
            Y = y,
            M = M, 
            K = K_spatial,
            verbose = False
        )
        stats = lmm.stats
        result = {
            'gene': gene,
            'pvalue': stats['pv20'].iloc[0],
            'lml_null': stats['lml0'].iloc[0],
            'lml_alt': stats['lml2'].iloc[0],
            'scale': stats['scale2'].iloc[0],
            'log_lr': stats['lml2'].iloc[0] - stats['lml0'].iloc[0]
        }
    except Exception as e:
        return {
            'gene': gene,
            'pvalu': np.nan,
            'lml_null': np.nan,
            'lml_alt': np.nan,
            'scale': np.nan,
            'log_lr': np.nan
        }


def optimize_length_scale_single_sample(
    sample_expression, 
    sample_tas_pred, 
    sample_coords,
    length_scales,
    n_genes_test = 20):

    print(f'Optimizing length scale using {n_genes_test} genes')

    n_genes_available = min(n_genes_test, sample_expression.shape[1])
    gene_indices = np.random.choice(
        sample_expression.shape[1],
        n_genes_available,
        replace = False
    )

    best_avg_ll = -np.inf
    best_length = None

    for length_scale in tqdm(length_scales, desc = 'Testing length scales'):
        K_spatial = fast_rbf_kernel(sample_coords, length_scale)
        K_spatial += 1e-3 * np.eye(K_spatial.shape[0]) # regularize

        # covariate matrix
        M = np.ones((len(sample_expression), 1))
        G = sample_tas_pred.reshape(-1, 1)

        avg_ll = 0
        valid_genes = 0

        for gene_idx in gene_indices:
            y = sample_expression.iloc[:, gene_idx].values.rehsape(-1, 1)
            lmm = limix.qtl.scan(
                G = np.zeros((len(sample_expression), 1)),
                Y = y,
                M = M,
                K = K_spatial,
                verbose = False
            )
            avg_ll += lmm.stats['lml0'].iloc[0]
            valid_genes += 1

        if valid_genes > 0:
            avg_ll /= valid_genes
            if avg_ll > best_avg_ll:
                best_avg_ll = avg_ll
                best_length = length_scale
    
    print(f'Optimal length scale: {best_length}')
    return best_length


def analyze_single_sample(
    sample_id,
    sample_metadata,
    sample_expression,
    sample_coords,
    length_scales,
    n_processes = 4,
    optimize_lengths = True
):
    print(f'Analyzing sample: {sample_id}')
    print(f'Number of genes: {sample_expression.shape[1]}')

    rs_pred_counts = sample_metadata['rs_prediction'].value_counts()
    print(f'RS Pred distribution: {dict(rs_pred_counts)}')

    if optimize_lengths:
        optimal_length = optimize_length_scale_single_sample(
            sample_expression, sample_metadata['rs_prediction'].values,
            sample_coords, length_scales
        )
    else:
        optimal_length = 140

    K_spatial = fast_rbf_kernel(sample_coords, optimal_length)
    K_spatial += 1e-3 * np.eye(K_spatial.shape[0])

    M = np.ones((len(sample_metadata), 1))
    G = sample_metadata['rs_prediction'].values.reshape(-1, 1)
    
    gene_data_list = []
    gene_list = list(sample_expression.columns)

    for gene in gene_list:
        y = sample_expression[gene].values.reshape(-1, 1)
        gene_data_list.append((
            gene, y, M, G, K_spatial
        ))

    print(f'Processing genes with {n_processes} processes')
    with mp.Pool(processes = n_processes) as pool:
        results_list = list(tqdm(
            pool.imap_unordered(process_single_sample_gene, gene_data_list),
            total = len(gene_data_list),
            desc = f'{sample_id} genes'
        ))
    
    sample_results = pd.DataFrame(results_list)
    sample_results = sample_results.dropna(subset = ['pvalue'])

    if len(sample_results) > 0:
        sample_results['padj'] = multipletests(sample_results['pvalue'], method = 'fdr_bh')[1]
        sample_results = sample_results(by = 'padj', ascending = True)
        sample_results['sample_id'] = sample_id
        sample_results['optimal_length_scale'] = optimal_length

        n_sig_05 = (sample_results['padj'] < 0.05).sum()
        n_sig_01 = (sample_results['padj'] < 0.01).sum()

        print(f'Results for {sample_id}')
        print(f'Genes tested: {len(sample_results)}')
        print(f'Significant at FDR < 0.05: {n_sig_05} ({n_sig_05/len(sample_results)*100:.1f}%)')
        print(f'Significant at FDR < 0.01: {n_sig_01} ({n_sig_01/len(sample_results)*100:.1f}%)')

    
    return sample_results
    

def run_within_sample_spatial_dge(
    metadata_path, 
    expression_matrix_path,
    coord_files,
    output_dir,
    n_processes = 4
):
    print('Within sample DGE!')
    os.makedirs(output_dir, exist_ok = True)

    print('loading data')
    metadata = pd.read_csv(metadata_path, index_col = 0)

    if 'barcode' in metadata.columns:
        metadata = metadata.set_index('barcode')

    expression_data = pd.read_csv(expression_matrix_path, index_col = 0)
    if 'barcode' in expression_data.columns:
        expression_data = expression_data.set_index('barcode')
    
    coords_dict = {}
    sample_ids = sorted(metadata['slide_name'].unique())

    for i, coord_file in enumerate(coord_files):
        if i < len(sample_ids):
            coords_df = pd.read_csv(coord_file, index_col = 0)
            coords_dict[sample_ids[i]] = coords_df[['x', 'y']].values
        
    
    all_sample_results = []
    sample_summaries = []

    length_scales = [20, 40, 60, 80, 100, 120]

    for sample_id in sample_ids:
        sample_mask = metadata['slide_name'] == sample_id
        sample_metadata = metadata[sample_mask]
        sample_expression = expression_data.loc[sample_metadata.index]

        if sample_id in coords_dict:
            sample_coords = coords_dict[sample_id]

            if len(sample_coords) == len(sample_metadata):
                sample_results = analyze_single_sample(
                    sample_id, sample_metadata, sample_expression,
                    sample_coords, length_scales, n_processes
                )

                if sample_results is not None:

                    all_sample_results.append(sample_results)
                    sample_output_path = os.path.join(output_dir, f'{sample_id}_results_optimized_length.csv')
                    sample_results.to_csv(sample_output_path, index = False)

                    summary = {
                        'sample_id': sample_id,
                        'n_spots': len(sample_metadata),
                        'n_genes_tested': len(sample_results),
                        'n_sig_05': (sample_results['padj'] < 0.05).sum(),
                        'n_sig_01': (sample_results['padj'] < 0.01).sum(),
                        'pct_sig_05': (sample_results['padj'] < 0.05).sum()/len(sample_results)*100,
                        'pct_sig_01': (sample_results['padj'] < 0.01).sum()/len(sample_results)*100,
                        'optimal_length_scale': sample_results['optimal_length_scale'].iloc[0],
                    }
                    sample_summaries.append(summary)

    if all_sample_results:
        combined_results = pd.concat(all_sample_results, ignore_index = True)
        combined_output_path = os.path.join(
            output_dir, 'all_samples_combined_results_optimized_length.csv'
        )
        combined_results.to_csv(combined_output_path, index = False)

        summary_df = pd.DataFrame(sample_summaries)
        summary_output_path = os.path.join(
            output_dir, 'sample_summary.csv'
        )
        summary_df.to_csv(summary_output_path, index = False)
        return combined_results, summary_df
    
    else:
        print('No samples successfully analyzed')
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type = str
    )
    parser.add_argument(
        '--output_dir', type = str
    )

    args = parser.parse_args()

    metadata_path = os.path.join(args.data_dir, 'metadata.csv')
    expression_matrix_path = os.path.join(args.data_dir, 'expression_matrix.csv')
    coord_files = sorted(glob.glob(os.path.join(data_dir, 'coords_*')))

    combined_results, summary_df = run_within_sample_spatial_dge(
        metadata_path = metadata_path,
        expression_matrix_path = expression_matrix_path,
        coord_files = coord_files,
        output_dir = output_dir,
        n_processes = 8
    )

