from datasets import get_dataset_miccai26


def get_dataset_tmi26(args, config):
    degraded_override = getattr(args, "degraded_path", "")
    if not degraded_override:
        return get_dataset_miccai26(args, config)

    if "MELA" in args.config:
        from datasets.MELA_temporal import MELADataset_Temporal

        gt_path = f"/workspace/data/pickle_data/MELA_GT_512_rmbed/{args.path_y}_rmbed.pickle"
        test_dataset = MELADataset_Temporal(args, degraded_override, gt_path)
        print(f"TMI26 dataset (MELA override, degraded_path={degraded_override})")
    elif "UHRCT" in args.config:
        from datasets.UHRCT import UHRCTDataset

        gt_path = f"/workspace/data/pickle_data/UHRCT_GT_512_rmbed/{args.path_y}.pickle"
        test_dataset = UHRCTDataset(args, degraded_override, gt_path)
        print(f"TMI26 dataset (UHRCT override, degraded_path={degraded_override})")
    else:
        raise ValueError(f"Unsupported config: {args.config}")

    dataset = None
    return dataset, test_dataset
