import os
import pickle
import numpy as np
import argparse


def main(pickle_file, numpy_file, new_prefix, PICKLE_FOLDER, PICKLE_FOLDER_LR, DDNM_FOLDER, NEW_PICKLE_FOLDER):
    with open(os.path.join(PICKLE_FOLDER, f"{pickle_file}.pickle"), 'rb') as f:
        data = pickle.load(f)

    with open(os.path.join(PICKLE_FOLDER_LR, f"{pickle_file}.pickle"), 'rb') as f:
        data_LR = pickle.load(f)

    # new_projections shape: (N,C,H,W) or (N,H,W)
    new_projections = np.load(os.path.join(DDNM_FOLDER, numpy_file, 'whole.npy'))

    old_projections = np.asarray(data['train']['projections'], dtype=np.float32)
    old_projections_lr = np.asarray(data_LR['train']['projections'], dtype=np.float32)
    if new_projections.shape[0] != old_projections.shape[0]:
        raise ValueError(
            f"Mismatch in number of projections: "
            f"Pickle has {old_projections.shape[0]}, "
            f"numpy has {new_projections.shape[0]}"
        )

    N = old_projections.shape[0]

    adjusted_projections = []
    for idx in range(N):
        old_proj = old_projections_lr[idx]   # (H,W)
        new_proj = new_projections[idx]       # (C,H,W) or (H,W)

        new_img = new_proj[0] if new_proj.ndim == 3 else new_proj

        min_val = old_proj.min()
        max_val = old_proj.max()

        scaled = new_img * (max_val - min_val) + min_val
        scaled = np.clip(scaled, 0.0, 1.0)

        adjusted_projections.append(scaled)

    adjusted_projections = np.stack(adjusted_projections, axis=0)  # (N,H,W)

    data['train']['projections'] = adjusted_projections

    os.makedirs(NEW_PICKLE_FOLDER, exist_ok=True)
    new_filename = os.path.join(NEW_PICKLE_FOLDER, f"{pickle_file}_{new_prefix}.pickle")
    with open(new_filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"New pickle file saved as: {new_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replace projections in a pickle file with DDNM-upsampled numpy projections."
    )
    parser.add_argument('--pickle', default='mela_0768',
                        help="Pickle base name (without .pickle extension)")
    parser.add_argument('--proj', default=None,
                        help="Folder name under --ddnm_folder that contains whole.npy")
    parser.add_argument('--prefix', default='DDNM',
                        help="Suffix appended to the output pickle filename")
    parser.add_argument('--pickle_folder', required=True, type=str,
                        help="Path to folder containing HR pickle files (e.g. ./data/pickle_data/CASE_UP/)")
    parser.add_argument('--pickle_folder_lr', required=True, type=str,
                        help="Path to folder containing LR pickle files (e.g. ./data/pickle_data/CASE_GT_LR/)")
    parser.add_argument('--new_pickle_folder', required=True, type=str,
                        help="Output folder for the new pickle files")
    parser.add_argument('--ddnm_folder', required=True, type=str,
                        help="Path to folder containing DDNM outputs (each subfolder must have whole.npy)")

    args = parser.parse_args()

    pickle_data_list = os.listdir(args.pickle_folder)
    proj_data_list = os.listdir(args.ddnm_folder)

    for pickle_data in pickle_data_list:
        for proj_data in proj_data_list:
            if pickle_data[:4] == 'init':
                continue
            if pickle_data[:-7] == proj_data:
                print("Current data:", pickle_data[:-7], proj_data)
                main(pickle_data[:-7], proj_data, args.prefix, args.pickle_folder, args.pickle_folder_lr, args.ddnm_folder, args.new_pickle_folder)
