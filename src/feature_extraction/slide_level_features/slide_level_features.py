import os
import numpy as np
import pandas as pd
import pickle as pkl
import scipy
from math import e
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix
from scipy.ndimage import label
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix
from scipy.stats import entropy
import networkx as nx
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import laplace


## pandas entropy function from https://stackoverflow.com/questions/49685591/how-to-find-the-entropy-of-each-column-of-data-set-by-python
def pandas_entropy(column, base=None):
    vc = pd.Series(column).value_counts(normalize=True, sort=False)
    base = e if base is None else base
    return -(vc * np.log(vc)/np.log(base)).sum()


def calculate_basic_stats(confident_set, stroma_preds_file, high_prob_thresh = 0.75):
    mean_pos_prob = np.mean(confident_set[confident_set['rs_pred'] == 1]['mean_pred'])
    mean_neg_prob = np.mean(confident_set[confident_set['rs_pred'] == 0]['mean_pred'])
    mean_prob = confident_set['mean_pred'].mean()

    median_pos_prob = confident_set[confident_set['rs_pred'] == 1]['mean_pred'].median()
    median_neg_prob = confident_set[confident_set['rs_pred'] == 0]['mean_pred'].median()
    median_prob = confident_set['mean_pred'].median()

    std_pos_prob = confident_set[confident_set['rs_pred'] == 1]['mean_pred'].std()
    std_neg_prob = confident_set[confident_set['rs_pred'] == 0]['mean_pred'].std()
    std_prob = confident_set['mean_pred'].std()

    entropy_pos_prob = pandas_entropy(confident_set[confident_set['rs_pred'] == 1]['mean_pred'])
    entropy_neg_prob = pandas_entropy(confident_set[confident_set['rs_pred'] == 0]['mean_pred'])
    entropy_prob = pandas_entropy(confident_set['mean_pred'])
    
    rs_pos_percent = len(confident_set[confident_set['rs_pred'] == 1])/len(stroma_preds_file)
    rs_neg_percent = len(confident_set[confident_set['rs_pred'] == 0])/len(stroma_preds_file)
    rs_pos_high_percent = len(confident_set[confident_set['mean_pred'] >= high_prob_thresh])/len(stroma_preds_file)
    rs_conf_percent = len(confident_set)/len(stroma_preds_file)
    
    basic_stats_feats = {
        'mean_pos_prob': mean_pos_prob, 
        'mean_neg_prob': mean_neg_prob, 
        'mean_prob': mean_prob, 
        'median_pos_prob': median_pos_prob,
        'median_neg_prob': median_neg_prob,
        'median_prob':median_prob,
        'std_pos_prob':std_pos_prob,
        'std_neg_prob':std_neg_prob,
        'std_prob':std_prob,
        'entropy_pos_prob':entropy_pos_prob,
        'entropy_neg_prob':entropy_neg_prob,
        'entropy_prob':entropy_prob,
        'rs_pos_percent':rs_pos_percent,
        'rs_neg_percent':rs_neg_percent,
        'rs_pos_high_percent':rs_pos_high_percent,
        'rs_conf_percent':rs_conf_percent
        }
    
    return basic_stats_feats


def nearest_neighbor_index(coords):
    tree = KDTree(coords)
    dists, _ = tree.query(coords, k = 2)
    mean_nn_dist = dists[:, 1].mean()
    n = len(coords)
    area = (coords[:, 0].max() - coords[:, 0].min()) * (coords[:, 1].max() - coords[:, 1].min())
    expected_dist = 0.5/np.sqrt(n/area)
    return mean_nn_dist/expected_dist


def inter_cluster_variance(centroids):
    d = pdist(centroids)
    return np.var(d)

def get_location_label(loc, mask):
    y, x = loc
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        return mask[y, x]
    return 0

def get_locations(slide_name, slide_path, mask_path, location_dict):
    locations = location_dict[location_dict['slide_name'] == slide_name]
    try:
        locations = locations[['x_loc', 'y_loc']].values
    except:
        locations = locations[['x', 'y']].values
    slide = openslide.OpenSlide(slide_path)
    mask = np.array(Image.open(mask_path))
    thumbnail = slide.get_thumbnail(mask.shape[::-1])
    downsample_rate = slide.level_dimensions[0][0] / mask.shape[1]
    locations_downsample = (locations / downsample_rate).astype("int")
    loc_labels = np.array(
        [get_location_label(loc, mask) for loc in locations_downsample]
    )
    return slide_name, thumbnail, mask, downsample_rate


def full_slide_features(confident_set, total_number_of_stroma_patches, connected_comp_size, hotspot_thresh = 0.75):
    patch_size = 256
    confident_set = confident_set.copy()
    try:
        confident_set["x_idx"] = (confident_set["x_loc"] // patch_size).astype(int)
        confident_set["y_idx"] = (confident_set["y_loc"] // patch_size).astype(int)
    except:
        confident_set["x_idx"] = (confident_set["x"] // patch_size).astype(int)
        confident_set["y_idx"] = (confident_set["y"] // patch_size).astype(int)


    x_min, y_min = confident_set["x_idx"].min(), confident_set["y_idx"].min()
    confident_set["x_grid"] = (confident_set["x_idx"] - x_min).copy()
    confident_set["y_grid"] = (confident_set["y_idx"] - y_min).copy()

    grid_w = confident_set["x_grid"].max() + 1
    grid_h = confident_set["y_grid"].max() + 1

    binary_grid = lil_matrix((grid_h, grid_w), dtype=bool)

    hotspot_mask = confident_set["mean_pred"] >= hotspot_thresh
    binary_grid[confident_set.loc[hotspot_mask, "y_grid"], confident_set.loc[hotspot_mask, "x_grid"]] = True
    
    binary_array = binary_grid.toarray().astype(bool)
    structure = np.ones((3, 3), dtype = int)
    labeled_array, num_regions = label(binary_array, structure = structure)
    num_hotspots = num_regions
    
    region_sizes = np.bincount(labeled_array.ravel())[1:]
    max_size = region_sizes.max()
    min_size = region_sizes.min()
    avg_size = region_sizes.mean()
    median_size = np.median(region_sizes)
    std_size = region_sizes.std()
    total_patches = len(hotspot_mask)
    percent_hotspot = hotspot_mask.sum()/total_number_of_stroma_patches
    
    centroids = []
    region_probs = []
    region_patch_counts = []
    for region_id in range(1, num_regions + 1):
        ys, xs = np.where(labeled_array == region_id)
        centroid_x = xs.mean() * patch_size + x_min * patch_size
        centroid_y = ys.mean() * patch_size + y_min * patch_size
        centroids.append((centroid_x, centroid_y))
        region_patch_counts.append(len(xs))
        region_probs.append([
            confident_set[(
                confident_set['x_grid'] == x) & (confident_set['y_grid'] == y) & (
                confident_set['mean_pred'] > hotspot_thresh)]['mean_pred'].values[0]
                         for x, y in zip(xs, ys)])
        
    centroids = np.array(centroids)
    centroid_std_x = np.std(centroids[:, 0])
    centroid_std_y = np.std(centroids[:, 1])
    
    distances = squareform(pdist(centroids))
    inter_hotspot_distances = distances[np.triu_indices(num_regions, k = 1)]
    min_dist = inter_hotspot_distances.min() if len(inter_hotspot_distances) > 0 else 0
    max_dist = inter_hotspot_distances.max() if len(inter_hotspot_distances) > 0 else 0
    avg_dist = inter_hotspot_distances.mean() if len(inter_hotspot_distances) > 0 else 0
    
    nni = nearest_neighbor_index(centroids)
    spread_var = inter_cluster_variance(centroids)
    
    total_area = grid_w * grid_h
    region_density = num_regions / total_area
    patch_density = hotspot_mask.sum() / total_area
    
    distances = distance_matrix(centroids, centroids)
    coalescence_radius = 10
    coalescence_counts = (distances < coalescence_radius).sum(axis=1) - 1  
    avg_coalescence = coalescence_counts.mean()
    
    np.fill_diagonal(distances, np.inf)
    region_merging_threshold = distances.min()
    avg_prob_in_region = np.mean([np.mean(r) for r in region_probs])
    prob_variance = np.mean([np.var(r) for r in region_probs])
    probs = confident_set.loc[hotspot_mask, 'mean_pred'].values
    hist, _ = np.histogram(probs, bins=20, range=(0, 1), density=False) #CODE CHANGE
    hist_probs = hist/hist.sum() # CODE CHANGE
    entropy_of_probs = entropy(hist_probs + 1e-6) # CODE CHANGE
    try:
        hotspot_coords = confident_set.loc[hotspot_mask, ['x_loc', 'y_loc']].values
    except:
        hotspot_coords = confident_set.loc[hotspot_mask, ['x', 'y']].values
    hotspot_probs = confident_set.loc[hotspot_mask, 'mean_pred'].values
    center_of_mass_x = np.average(hotspot_coords[:, 0], weights=hotspot_probs)
    center_of_mass_y = np.average(hotspot_coords[:, 1], weights=hotspot_probs)

    rs_probs = confident_set['mean_pred'].values
    try:
        coords = confident_set[['x_loc', 'y_loc']].values
    except:
        coords = confident_set[['x', 'y']].values

    nbrs = NearestNeighbors(n_neighbors=6).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    local_contrasts = []
    for i, idx in enumerate(indices):
        neighbors = rs_probs[idx[1:]]  # exclude self
        contrast = np.abs(rs_probs[i] - neighbors.mean())
        local_contrasts.append(contrast)

    mean_contrast = np.mean(local_contrasts)
    std_contrast = np.std(local_contrasts)
    high_contrast_percent = np.mean(np.array(local_contrasts) > 0.2)

    laplace_vals = None
    if grid_h > 1 and grid_w > 1:
        rs_grid = np.zeros((grid_h, grid_w), dtype=float)
        for _, row in confident_set.iterrows():
            rs_grid[int(row['y_grid']), int(row['x_grid'])] = row['mean_pred']
        lap = laplace(rs_grid)
        lap_vals = lap[binary_array]
        laplace_mean = np.mean(np.abs(lap_vals))
        laplace_std = np.std(np.abs(lap_vals))
    else:
        laplace_mean = 0
        laplace_std = 0
        
    features = {
        'max_size': max_size,
        'min_size':min_size,
        'avg_size' :avg_size, 
        'median_size' :median_size,
        'std_size' :std_size, 
        'total_patches':total_patches,
        'percent_hotspot' :percent_hotspot, 
        'min_dist' :min_dist,
        'max_dist' :max_dist, 
        'avg_dist' :avg_dist,
        'nni': nni, 
        'spread_var':spread_var,
        'region_density':region_density,
        'patch_density':patch_density,
        'avg_coalescence':avg_coalescence,
        'avg_prob_in_region':avg_prob_in_region,
        'prob_variance':prob_variance,
        'entropy_of_probs':entropy_of_probs,
        'center_of_mass_x':center_of_mass_x,
        'center_of_mass_y':center_of_mass_y,
        'num_hotspots':num_hotspots,
        'mean_contrast':mean_contrast,
        'std_contrast':std_contrast,
        'high_contrast_percent':high_contrast_percent,
        'laplace_mean':laplace_mean,
        'laplace_std':laplace_std,
    }
    
    return features, region_patch_counts, region_probs, centroids


def get_graph_feats(confident_set, centroids, proximity_threshold = 2000):
    G = nx.Graph()
    for i, (x,y) in enumerate(centroids):
        G.add_node(i, pos = (x, y))
    

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist <= proximity_threshold:
                G.add_edge(i, j, weight = dist)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = np.mean([deg for _, deg in G.degree()])
    clustering_coeff = nx.average_clustering(G)
    if nx.is_connected(G):
        graph_diameter = nx.diameter(G)
        avg_shortest_path = nx.average_shortest_path_length(G)

    else:
        largest_cc = max(nx.connected_components(G), key = len)
        subgraph = G.subgraph(largest_cc)
        graph_diameter = nx.diameter(subgraph)
        avg_shortest_path = nx.average_shortest_path_length(subgraph)

    G = nx.Graph()
    for i, (x, y) in enumerate(centroids):
        G.add_node(i, pos=(x, y))

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist <= proximity_threshold:  # same threshold as before
                G.add_edge(i, j, weight=dist)

    if G.number_of_nodes() > 1:
        centrality = nx.betweenness_centrality(G, weight='weight')
        cent_vals = np.array(list(centrality.values()))
        mean_betweenness = cent_vals.mean()
        max_betweenness = cent_vals.max()
        std_betweenness = cent_vals.std()
        high_betweenness_count = (cent_vals > 0.1).sum()
    else:
        mean_betweenness = max_betweenness = std_betweenness = high_betweenness_count = 0

    topo = {
        'num_nodes':num_nodes,
        'num_edges':num_edges,
        'avg_degree':avg_degree,
        'clustering_coeff':clustering_coeff,
        'graph_diameter':graph_diameter,
        'avg_shortest_path':avg_shortest_path,
        'mean_betweenness':mean_betweenness,
        'max_betweenness':max_betweenness,
        'std_betweenness':std_betweenness,
        'high_betweenness_count': high_betweenness_count
    }
    return topo


def extract_distance_feats(df, distance_col = 'distance_bin_mm', rs_col = 'mean_pred',
    area_col = None, near_mm = 1, far_from_mm = 3, eps = 1e-8):

    if df.empty:
        return {}

    g = df.groupby(distance_col, as_index = True)
    count_r = g.size().rename('n').astype(float)
    sum_rs_r = g[rs_col].sum().rename('s')
    if area_col is not None and (area_col in df.columns):
        area_r = g[area_col].sum().rename('A')
        sum_rs_area_r = g.apply(lambda x: np.sum(x[rs_col] * x[area_col])).rename('sA')
    else:
        area_r = None
        sum_rs_area_r = None

    w_r = sum_rs_area_r if sum_rs_area_r is not None else sum_rs_r
    w_r = w_r.fillna(0.0)

    q_r = (sum_rs_r/(count_r + eps)).rename('q') # mean rs per patch in each bin

    d_bins = w_r.index.values.astype(float)
    d_centers = d_bins

    total_w = float(w_r.sum())
    p_r = (w_r / (total_w + eps))

    near_mask = d_centers <= near_mm
    far_mask = d_centers >= far_from_mm

    near_mass = float(w_r[near_mask].sum())
    far_mass = float(w_r[far_mask].sum())

    edge_core_ratio = near_mass / (far_mass + eps)
    com = float(np.sum(d_centers * p_r))
    entropy = float(-np.sum(p_r * np.log(p_r + eps)))

    slope = float(np.polyfit(d_centers, np.log(q_r.values + eps), 1)[0]) if len(q_r) >= 2 else np.nan
    cdf = np.cumsum(p_r.values[np.argsort(d_centers)])
    d_sorted = np.sort(d_centers)
    half_idx = int(np.searchsorted(cdf, 0.5, side = 'left')) if total_w > 0 else 0
    half_distance = float(d_sorted[min(half_idx, len(d_sorted) - 1)]) if len(d_sorted) else np.nan


    feats = {
        'rs_edge_core_ratio': edge_core_ratio,
        'rs_com_distance': com, # center of mass
        'rs_distance_entropy': entropy,
        'rs_lograte_slope': slope, 
        'rs_half_distance': half_distance,
        'rs_total_mass': total_w,
        'rs_mean_rate_over_bins': float(q_r.mean() if len(q_r) else np.nan),
        'rs_max_rate_over_bins': float(q_r.max() if len(q_r) else np.nan)
    }

    for d, q in q_r.items():
        feats[f"rs_rate_bin_{int(d)}mm"] = float(q)

    for d, n in count_r.items():
        feats[f"patch_count_bin_{int(d)}mm"] = float(n)

    if area_r is not None:
        for d, A in area_r.items():
            feats[f"tissue_area_bin_{int(d)}mm"] = float(A)

    return feats

def aggregate_ring_features(patch_df, tau = 0.7, q_list = [0.75], min_ring_n = 1):
    def _agg(group):
        vals = group['mean_pred'].values
        out = {}
        out['mean_rs'] = float(np.mean(vals)) if len(vals) else np.nan
        for q in q_list:
            out[f"q{int(q*100)}_rs"] = float(np.quantile(vals, q)) if len(vals) else np.nan
        out['burden_gt_tau'] = float(np.mean(vals > tau)) if len(vals) else np.nan
        out['patch_count'] = int(len(vals))

        return pd.Series(out)

    g = patch_df.groupby(['slide_name', 'ring_idx'], observed = True).apply(_agg)
    for col in [c for c in g.columns if c != 'patch_count']:
        g.loc[g['patch_count'] < min_ring_n, col] = np.nan

    return g


def extract_ring_features(patch_df, tau, q_list, min_ring_n, ring_labels):
    ring_feat_df = aggregate_ring_features(patch_df, tau = tau, q_list = q_list, 
        min_ring_n = min_ring_n)

    wide_parts = []
    for feat in [c for c in ring_feat_df if c != 'patch_count']:
        tmp = ring_feat_df[[feat]].unstack(level = 1)
        tmp.columns = [f'{feat}__{ring_labels[int(idx)]}' for idx in tmp.columns.get_level_values(1)]
        wide_parts.append(tmp)

    wide_df = pd.concat(wide_parts, axis = 1)
    pc = ring_feat_df[['patch_count']].unstack(level = 1)
    pc.columns = [f"patch_count__{ring_labels[int(idx)]}" for idx in pc.columns.get_level_values(1)]
    wide_df = pd.concat([wide_df, pc], axis = 1)
    return wide_df
