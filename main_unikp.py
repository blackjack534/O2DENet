from models import MESI
from time import time
from utils import set_seed, graph_collate_func
from configs import get_cfg_defaults
from dataloader import ESIDataset
from torch.utils.data import DataLoader
from run.trainer import Trainer
from run.trainer_DDP import Trainer_DDP
from run.trainer_reg import Trainer_Reg
from run.trainer_reg_DDP import Trainer_Reg_DDP

import torch
import argparse
import warnings, os, re
import pandas as pd
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from GraphAug import DGLGraphFeatureMaskAug

# Stage2
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


def create_experiment_directory(result, dataset):
    if dist.is_initialized():
        rank = dist.get_rank()
        is_main_process = (rank == 0)
    else:
        is_main_process = True

    base_path = os.path.join(result, dataset)
    if not os.path.exists(base_path):
        try:
            os.makedirs(base_path)
        except FileExistsError:
            pass

    if dist.is_initialized():
        dist.barrier()

    exp_folders = [d for d in os.listdir(base_path) if d.startswith('exp') and os.path.isdir(os.path.join(base_path, d))]
    exp_numbers = [int(re.search(r'exp(\d+)', folder).group(1)) for folder in exp_folders if re.search(r'exp(\d+)', folder)]
    max_exp_number = max(exp_numbers) if exp_numbers else -1
    new_exp_number = max_exp_number + 1
    new_exp_folder = f'exp{new_exp_number}'
    new_exp_path = os.path.join(base_path, new_exp_folder)

    if is_main_process:
        if not os.path.exists(new_exp_path):
            os.makedirs(new_exp_path)
        os.system(f'cp -r ./module {new_exp_path}/')
        os.system(f'cp ./models.py {new_exp_path}/')

    if dist.is_initialized():
        dist.barrier()
    return new_exp_path


def init_device_and_ddp():
    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    return device, local_rank


def _get_rank_and_main():
    if dist.is_initialized():
        rank = dist.get_rank()
        return rank, (rank == 0)
    return 0, True


def _named_modules_dict(model):
    return dict(model.named_modules())


class FeatureExtractorHook:
    """
    Grab intermediate tensor from a named submodule via forward hook.
    It stores the last batch feature in self.last_feat.
    """
    def __init__(self, model, module_name: str):
        modules = _named_modules_dict(model)
        if module_name not in modules:
            raise ValueError(
                f"[FeatureHook] module '{module_name}' not found in model.named_modules(). "
                f"Available examples: {list(modules.keys())[:30]} ..."
            )
        self.module = modules[module_name]
        self.last_feat = None
        self.hook = self.module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, out):
        # out can be tuple/list/tensor
        if isinstance(out, (tuple, list)):
            out = out[0]
        # ensure tensor
        if torch.is_tensor(out):
            self.last_feat = out.detach()
        else:
            self.last_feat = None

    def close(self):
        self.hook.remove()


@torch.no_grad()
def extract_split_features(model, dataloader, device, feat_hook: FeatureExtractorHook, split_name: str, save_dir: str):
    """
    Run model forward on dataloader, collect features + labels, save to per-rank file.
    NOTE: assumes ESIDataset returns (graph, y, ...) compatible with your model forward.
    """
    rank, _ = _get_rank_and_main()
    model.eval()

    feats = []
    ys = []

    for batch in dataloader:
        # 这里按你原本 trainer 的 batch 形式来：一般 graph_collate_func 会返回 (batched_graph, labels, ...)
        # 你需要根据实际 batch 结构调整下面两行解包。
        if isinstance(batch, (tuple, list)):
            g = batch[0]
            y = batch[1]
        else:
            raise RuntimeError("Unexpected batch type; please adapt unpacking logic.")

        # 将输入送入 device（DGLGraph 通常有自己的 to(device)）
        try:
            g = g.to(device)
        except Exception:
            pass

        if torch.is_tensor(y):
            y = y.to(device)

        # 触发 forward（输出本身不重要，重要的是 hook 抓到的中间特征）
        _ = model(g)

        if feat_hook.last_feat is None:
            raise RuntimeError(
                f"[FeatureHook] No feature captured. "
                f"Check --feat_module points to a layer that outputs a tensor."
            )

        f = feat_hook.last_feat
        # flatten to (B, D)
        if f.dim() > 2:
            f = torch.flatten(f, start_dim=1)

        feats.append(f.cpu().numpy())
        ys.append(y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y))

    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, 0), dtype=np.float32)
    Y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.float32)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{split_name}_rank{rank}.npz")
    np.savez_compressed(out_path, X=X, y=Y)
    return out_path


def merge_rank_features(save_dir: str, split_name: str):
    """
    Merge per-rank npz into a single npz (rank0 only).
    """
    rank, is_main = _get_rank_and_main()
    if dist.is_initialized():
        dist.barrier()

    if not is_main:
        if dist.is_initialized():
            dist.barrier()
        return None

    files = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.startswith(split_name) and f.endswith(".npz")]
    if not files:
        raise RuntimeError(f"No files found to merge for split={split_name} in {save_dir}")

    Xs, Ys = [], []
    for fp in sorted(files):
        data = np.load(fp)
        Xs.append(data["X"])
        Ys.append(data["y"])

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(Ys, axis=0)

    merged_path = os.path.join(save_dir, f"{split_name}.npz")
    np.savez_compressed(merged_path, X=X, y=y)

    if dist.is_initialized():
        dist.barrier()
    return merged_path


def load_split_npz(save_dir: str, split_name: str):
    data = np.load(os.path.join(save_dir, f"{split_name}.npz"))
    return data["X"], data["y"]


def stage2_train_extratrees(feature_dir: str, output_dir: str):
    X_train, y_train = load_split_npz(feature_dir, "train")
    X_val, y_val = load_split_npz(feature_dir, "val")
    X_test, y_test = load_split_npz(feature_dir, "test")

    # ExtraTreesRegressor: 常用稳健默认值（你也可以外部传参）
    reg = ExtraTreesRegressor(
        n_estimators=1000,
        random_state=0,
        n_jobs=-1,
        max_features="auto",
        min_samples_leaf=1,
    )
    reg.fit(X_train, y_train)

    def eval_split(name, X, y):
        pred = reg.predict(X)
        rmse = mean_squared_error(y, pred, squared=False)
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        print(f"[ExtraTrees][{name}] RMSE={rmse:.6f}  MAE={mae:.6f}  R2={r2:.6f}")
        return {"split": name, "rmse": rmse, "mae": mae, "r2": r2}

    metrics = []
    metrics.append(eval_split("val", X_val, y_val))
    metrics.append(eval_split("test", X_test, y_test))

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(reg, os.path.join(output_dir, "extratrees.joblib"))
    pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "extratrees_metrics.csv"), index=False)
    print(f"[ExtraTrees] Saved model/metrics to: {output_dir}")


def build_datasets(cfg, task):
    dataFolder = cfg.SOLVER.DATA
    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 你原来的逻辑：train = train+val, val = test
    df_train = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    df_val = df_test

    graph_aug = DGLGraphFeatureMaskAug(
        atom_mask_ratio=0.05,
        bond_mask_ratio=0.10,
        protect_core=True,
        mask_value=0.0,
        inplace=False
    )

    train_dataset = ESIDataset(
        df_train.index.values, df_train, task,
        graph_aug=graph_aug,
        graph_aug_on_orig=False,
        graph_aug_on_aug=True
    )
    val_dataset = ESIDataset(
        df_val.index.values, df_val, task,
        graph_aug=graph_aug,
        graph_aug_on_orig=False,
        graph_aug_on_aug=True
    )
    test_dataset = ESIDataset(
        df_test.index.values, df_test, task,
        graph_aug=graph_aug,
        graph_aug_on_orig=False,
        graph_aug_on_aug=True
    )
    return train_dataset, val_dataset, test_dataset


def build_loaders(cfg, train_dataset, val_dataset, test_dataset, for_feature_extraction=False):
    """
    for_feature_extraction=True 时，建议不 shuffle，drop_last=False
    """
    if torch.cuda.device_count() > 1:
        params = {
            'batch_size': cfg.SOLVER.BATCH_SIZE,
            'shuffle': False,
            'num_workers': cfg.SOLVER.NUM_WORKERS,
            'drop_last': False if for_feature_extraction else True,
            'collate_fn': graph_collate_func
        }
        train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset, shuffle=not for_feature_extraction), **params)
        val_loader = DataLoader(val_dataset, sampler=DistributedSampler(val_dataset, shuffle=False), **params)
        test_loader = DataLoader(test_dataset, sampler=DistributedSampler(test_dataset, shuffle=False), **params)
    else:
        params = {
            'batch_size': cfg.SOLVER.BATCH_SIZE,
            'shuffle': False if for_feature_extraction else True,
            'num_workers': cfg.SOLVER.NUM_WORKERS,
            'drop_last': False if for_feature_extraction else True,
            'collate_fn': graph_collate_func
        }
        train_loader = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        val_loader = DataLoader(val_dataset, **params)
        test_loader = DataLoader(test_dataset, **params)

    return train_loader, val_loader, test_loader


def main():
    device, local_rank = init_device_and_ddp()
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    parser = argparse.ArgumentParser(description="Two-stage: MESI feature extraction + ExtraTreesRegressor")
    parser.add_argument('--model', required=True, help="path to model config file", type=str)
    parser.add_argument('--data', required=True, help="path to data config file", type=str)
    parser.add_argument('--task', required=True, help="task type: regression, binary", choices=['regression', 'binary'], type=str)

    # NEW
    parser.add_argument('--stage', default='all', choices=['1', '2', 'all'], help="Run stage 1 (MESI->features), stage 2 (ExtraTrees), or both")
    parser.add_argument('--feat_module', default='encoder', type=str,
                        help="Name of the submodule to hook for features, e.g., encoder/gnn/projector")
    parser.add_argument('--feature_dir', default=None, type=str,
                        help="Directory to save/load extracted features. If None, will use output_dir/features")

    # 可选：stage1 是否训练
    parser.add_argument('--skip_train', action='store_true',
                        help="Stage1: skip training and only load checkpoint then extract features")
    parser.add_argument('--checkpoint', default="/root/autodl-tmp/OmniESI-main/results/CatPred_kcat/MESI/best_model_epoch.pth", type=str)

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.model)
    cfg.merge_from_file(args.data)
    set_seed(cfg.SOLVER.SEED[0])

    print(f"Model Config: {args.model}")
    print(f"Data Config: {args.data}")
    print(f"Task: {args.task}")
    print(f"Stage: {args.stage}")
    print(f"Running on: {device}", end="\n\n")

    # Create output exp folder
    output_dir = create_experiment_directory(cfg.RESULT.OUTPUT_DIR, cfg.SOLVER.SAVE)
    if args.feature_dir is None:
        feature_dir = os.path.join(output_dir, "features")
    else:
        feature_dir = args.feature_dir

    rank, is_main = _get_rank_and_main()
    if is_main:
        os.system(f'cp -r {args.data} {output_dir}/')
        os.system(f'cp -r {args.model} {output_dir}/')

    # Build datasets/loaders
    train_dataset, val_dataset, test_dataset = build_datasets(cfg, args.task)

    # Stage 1: train/load MESI and extract features
    if args.stage in ['1', 'all']:
        # loaders for training
        train_loader, val_loader, test_loader = build_loaders(cfg, train_dataset, val_dataset, test_dataset, for_feature_extraction=False)

        model = MESI(**cfg)
        model.to(device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
            )

        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        torch.backends.cudnn.benchmark = True

        # Initialize Trainer
        if torch.cuda.device_count() > 1:
            if args.task == 'binary':
                trainer = Trainer_DDP(model, opt, device, train_loader, val_loader, test_loader,
                                     val_sampler=DistributedSampler(val_dataset),
                                     test_sampler=DistributedSampler(test_dataset),
                                     output=output_dir, **cfg)
            else:
                trainer = Trainer_Reg_DDP(model, opt, device, train_loader, val_loader, test_loader,
                                         val_sampler=DistributedSampler(val_dataset),
                                         test_sampler=DistributedSampler(test_dataset),
                                         output=output_dir, **cfg)
        else:
            if args.task == 'binary':
                trainer = Trainer(model, opt, device, train_loader, val_loader, test_loader, output=output_dir, **cfg)
            else:
                trainer = Trainer_Reg(model, opt, device, train_loader, val_loader, test_loader, output=output_dir, **cfg)

        # Load checkpoint
        if os.path.exists(args.checkpoint):
            # DDP: trainer.model might be DDP wrapper; load into underlying module if needed
            try:
                trainer.model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
            except Exception:
                if hasattr(trainer.model, "module"):
                    trainer.model.module.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
                else:
                    raise
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            print(f"Checkpoint not found: {args.checkpoint}")

        if not args.skip_train:
            _ = trainer.train()

        # ====== Feature extraction loaders (no shuffle, no drop_last)
        fe_train_loader, fe_val_loader, fe_test_loader = build_loaders(cfg, train_dataset, val_dataset, test_dataset, for_feature_extraction=True)

        # Hook feature module: if DDP wrapped, hook underlying module
        hook_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        feat_hook = FeatureExtractorHook(hook_model, args.feat_module)

        # Extract per-rank
        extract_split_features(hook_model, fe_train_loader, device, feat_hook, "train", feature_dir)
        extract_split_features(hook_model, fe_val_loader, device, feat_hook, "val", feature_dir)
        extract_split_features(hook_model, fe_test_loader, device, feat_hook, "test", feature_dir)

        feat_hook.close()

        # Merge rank features
        merge_rank_features(feature_dir, "train")
        merge_rank_features(feature_dir, "val")
        merge_rank_features(feature_dir, "test")

        if is_main:
            print(f"[Stage1] Features saved to: {feature_dir}")

    # Stage 2: train ExtraTreesRegressor on extracted features
    if args.stage in ['2', 'all']:
        rank, is_main = _get_rank_and_main()
        # ExtraTrees 在 DDP 下只需要 rank0 跑即可
        if is_main:
            stage2_out = os.path.join(output_dir, "extratrees")
            stage2_train_extratrees(feature_dir, stage2_out)
        if dist.is_initialized():
            dist.barrier()

    if is_main:
        print(f"Directory for saving result: {output_dir}")
    return output_dir


if __name__ == '__main__':
    s = time()
    out = main()
    e = time()
    rank, is_main = _get_rank_and_main()
    if is_main:
        print(f"Total running time: {round(e - s, 2)}s")