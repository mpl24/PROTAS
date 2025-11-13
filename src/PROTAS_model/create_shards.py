import os
import math
import pandas as pd
import numpy as np
from tqdm import tqdm

def create_shards(
    dataframe, 
    biopsy_data_root, 
    rp_data_root, 
    shard_size = 50000, 
    out_dir = 'train'
    ):
    total_files = dataframe.shape[0]
    num_shards = math.ceil(total_files/shard_size)

    print(f'Total number of samples: {total_files}, creating {num_shards}, with up to {shard_size} samples each')
    mapping_bx_rp = {
        'biopsy': 0,
        'rp': 1
    }
    start_idx = 0
    for shard_idx in tqdm(range(num_shards)):
        end_idx = min(start_idx + shard_size, total_files)
        current_df = dataframe.iloc[start_idx:end_idx]
        chunk_len = len(current_df)

        shard_data_list = []
        label_data_list = []
        domain_data_list = []
        for ind, row in tqdm(current_df.iterrows(), total = len(current_df)):
            domain = row['domain']
            x, y = row['x'], row['y']
            slide_name = row['slide_name']
            label = row['label']
            
            if domain == 'rp':
                data_path = os.path.join(rp_data_root, slide_name, f'{x}_{y}.npy')
                try:
                    sample_arr = np.load(data_path)
                except:
                    print(f'Issue with data path: {data_path}')
                    continue
            elif domain == 'biopsy':
                data_path = os.path.join(biopsy_data_root, slide_name, f'{x}_{y}.npy')
                try:
                    sample_arr = np.load(data_path)
                except:
                    print(f'Issue with data path: {data_path}')
                    continue
            else:
                continue

            if sample_arr is None:
                continue
            if len(sample_arr.shape) != 1:
                continue

            shard_data_list.append(sample_arr)
            label_data_list.append(label)
            domain_data_list.append(mapping_bx_rp[domain])
        
        shard_data = np.stack(shard_data_list, axis = 0)
        shard_path = os.path.join(out_dir, f'shard_{shard_idx}.npy')
        np.save(shard_path, shard_data)

        label_data = np.stack(label_data_list)
        label_path = os.path.join(out_dir, f'labels_{shard_idx}.npy')
        np.save(label_path, label_data)
        
        domain_data = np.stack(domain_data_list)
        domain_path = os.path.join(out_dir, f'domains_{shard_idx}.npy')
        np.save(domain_path, domain_data)
        
        del shard_data_list
        del current_df
        del label_data_list
        del domain_data_list

        start_idx = end_idx


def main(args):
    save_folder = os.path.join(args.save_root, 'test')
    os.makedirs(save_folder, exist_ok = True)

    train_df = pd.read_csv(args.train_df_file, index_col = 0)
    train_df = train_df.sample(len(train_df), random_state = 24).reset_index(drop = True)

    create_shards(
        train_df,
        biopsy_data_root = args.biopsy_data_root,
        rp_data_root = args.rp_data_root,
        out_dir = save_folder
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_root', type = str
    )
    parser.add_argument(
        '--train_df_file', type = str, desc = 'File with training info - x, y, label, slide_name etc'
    )
    parser.add_argument(
        '--biopsy_data_root', type = str, desc = 'Folder with all uni embeds for biopsy stroma patches'
    )
    parser.add_argument(
        '--rp_data_root', type = str, desc = 'Folder with all uni embeds for rp stroma patches'
    )
    main(args)