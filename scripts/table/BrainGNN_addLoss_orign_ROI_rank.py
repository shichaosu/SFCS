# Prediction or table-preparation helper for the SFCS workflow.
# Update the path variables below before running this script.
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import random

FEATURE_DIR = "data/data_use/Final_data/GLM_Sparsify_Pos/20Coefficient/CN/"
SC_DIR = "data/SC/aparc/SC_Std0_1_npy/"
TABLE_PATH = "data/data_use/table/filtered_assessment_results.xlsx"
OUTPUT_DIR = "data/data_use/table/result_CN/trans/Pos/20Coefficient/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_CORES = 128
BATCH_SIZE_REAL = 32
BATCH_SIZE_PERM = 64
EPOCHS_REAL = 60
EPOCHS_PERM = 10
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
N_FOLDS_REAL = 5
N_FOLDS_PERM = 3
N_PERMUTATIONS = 1000
EARLY_STOP_PATIENCE_PERM = 3
EARLY_STOP_MIN_DELTA_PERM = 1e-4
SEED = 42
LAMBDA_AUX = 0.1
EPS = 1e-15
ROI_NUM = 68

torch.set_num_threads(CPU_CORES)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(SEED)


def sanitize_filename(value):
    """Script documentation for this SFCS analysis step."""
    safe = re.sub(r"[\\/]+", "_", str(value))
    safe = re.sub(r"\s+", "_", safe).strip("_")
    return safe or "unknown"


def topk_loss(s, ratio):
    """Script documentation for this SFCS analysis step."""
    if ratio > 0.5:
        ratio = 1 - ratio
    s = s.sort(dim=1).values

    k = int(s.size(1) * ratio)
    if k > 0:
        loss_low = -torch.log(1 - s[:, :k] + EPS).mean()
        loss_high = -torch.log(s[:, -k:] + EPS).mean()
        return loss_low + loss_high
    else:
        return torch.tensor(0.0, device=s.device)

        # ================= CCC Loss =================


class CCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CCCLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        mean_pred = torch.mean(y_pred)
        mean_true = torch.mean(y_true)
        var_pred = torch.var(y_pred, unbiased=False)
        var_true = torch.var(y_true, unbiased=False)
        covar = torch.mean((y_pred - mean_pred) * (y_true - mean_true))
        numerator = 2 * covar
        denominator = var_pred + var_true + (mean_pred - mean_true) ** 2 + self.eps
        ccc = numerator / denominator
        return 1 - ccc

        # ================= Dataset =================


class BrainDataset(Dataset):
    def __init__(self, features, adjacencies, labels):
        self.features = torch.FloatTensor(features)
        self.adjacencies = torch.FloatTensor(adjacencies)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.adjacencies[idx], self.labels[idx]


class ROIAwareConv(nn.Module):
    def __init__(self, in_dim, out_dim, n_nodes=ROI_NUM):
        super(ROIAwareConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_nodes = n_nodes
        self.register_buffer("pos_encoding", torch.eye(n_nodes))
        self.weight_generator = nn.Sequential(
            nn.Linear(n_nodes, 32), nn.ReLU(), nn.Linear(32, in_dim * out_dim)
        )
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj):
        batch_size = x.size(0)
        pos = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        weights = self.weight_generator(pos)
        weights = weights.view(batch_size, self.n_nodes, self.in_dim, self.out_dim)
        x_agg = torch.matmul(adj, x)
        out = torch.einsum("bni,bnio->bno", x_agg, weights)
        return out + self.bias


class TopKPooling(nn.Module):
    def __init__(self, in_dim, ratio=0.5):
        super(TopKPooling, self).__init__()
        self.ratio = ratio
        self.scorer = nn.Linear(in_dim, 1)

    def forward(self, x, current_mask=None):
        loss_unit = (torch.norm(self.scorer.weight, p=2) - 1) ** 2

        score = self.scorer(x).squeeze(-1)  # (B, N)

        if current_mask is not None:
            score = score.masked_fill(current_mask.squeeze(-1) == 0, -1e9)

        score_sig = torch.sigmoid(score)
        loss_pool = topk_loss(score_sig, self.ratio)
        k = int(x.size(1) * self.ratio)
        topk_values, topk_indices = torch.topk(score_sig, k, dim=1)

        threshold = topk_values[:, -1].unsqueeze(1)
        new_mask = (score_sig >= threshold).float().unsqueeze(-1)

        if current_mask is not None:
            new_mask = new_mask * current_mask
        return x * new_mask, new_mask, loss_pool, loss_unit, score_sig


class BrainGNN(nn.Module):
    def __init__(self, n_nodes=ROI_NUM, in_dim=24, hidden_dim=16, out_dim=1):
        super(BrainGNN, self).__init__()

        # Block 1
        self.conv1 = ROIAwareConv(in_dim, hidden_dim, n_nodes)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = TopKPooling(hidden_dim, ratio=0.5)

        # Block 2
        self.conv2 = ROIAwareConv(hidden_dim, hidden_dim, n_nodes)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.pool2 = TopKPooling(hidden_dim, ratio=0.5)

        # Readout
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_nodes * hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x, adj):
        # --- Block 1 ---
        x = self.conv1(x, adj)
        x = F.leaky_relu(x)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        x, mask1, lp1, lu1, score1 = self.pool1(x)

        # --- Block 2 ---
        x = self.conv2(x, adj)
        x = F.leaky_relu(x)
        x = x.permute(0, 2, 1)
        x = self.bn2(x)
        x = x.permute(0, 2, 1)

        x = x * mask1
        x, mask2, lp2, lu2, _ = self.pool2(x, current_mask=mask1)

        out = self.fc_block(x)
        aux_loss = lp1 + lu1 + lp2 + lu2
        return out.squeeze(-1), aux_loss, score1


def load_data_tensor(folder_id):
    feat_path = os.path.join(FEATURE_DIR, f"{folder_id }.npy")
    sc_path = os.path.join(SC_DIR, f"{folder_id }.npy")
    if not os.path.exists(feat_path) or not os.path.exists(sc_path):
        return None, None
    feat = np.load(feat_path)
    sc = np.load(sc_path)
    if feat.shape != (68, 24) or sc.shape != (68, 68):
        return None, None
    return feat, sc


def run_cross_validation(
    X_feats,
    X_adjs,
    Y_labels,
    IDs,
    epochs=EPOCHS_REAL,
    n_folds=N_FOLDS_REAL,
    batch_size=BATCH_SIZE_REAL,
    compute_importance=True,
    early_stop=False,
    early_stop_patience=EARLY_STOP_PATIENCE_PERM,
    early_stop_min_delta=EARLY_STOP_MIN_DELTA_PERM,
    verbose=True,
):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    y_pred_all = np.zeros_like(Y_labels)
    y_true_all = np.zeros_like(Y_labels)
    fold_importances = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_feats)):
        if verbose:
            print(f"  Fold {fold +1 }/{n_folds }", end="\r")

        X_train, X_test = X_feats[train_idx], X_feats[test_idx]
        Y_train, Y_test = Y_labels[train_idx], Y_labels[test_idx]

        N_train, N_nodes, N_dim = X_train.shape
        N_test = X_test.shape[0]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, N_dim)).reshape(
            N_train, N_nodes, N_dim
        )
        X_test_scaled = scaler.transform(X_test.reshape(-1, N_dim)).reshape(
            N_test, N_nodes, N_dim
        )

        train_dataset = BrainDataset(X_train_scaled, X_adjs[train_idx], Y_train)
        test_dataset = BrainDataset(X_test_scaled, X_adjs[test_idx], Y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        model = BrainGNN(n_nodes=ROI_NUM, in_dim=24).to(DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = CCCLoss().to(DEVICE)

        # Train
        best_loss = float("inf")
        bad_epochs = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for feats, adjs, labels in train_loader:
                feats, adjs, labels = (
                    feats.to(DEVICE),
                    adjs.to(DEVICE),
                    labels.to(DEVICE),
                )
                optimizer.zero_grad()
                preds, aux_loss, _ = model(feats, adjs)
                main_loss = criterion(preds, labels)
                loss = main_loss + LAMBDA_AUX * aux_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if early_stop and n_batches > 0:
                avg_loss = epoch_loss / n_batches
                if avg_loss < best_loss - early_stop_min_delta:
                    best_loss = avg_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= early_stop_patience:
                        break

                        # Val
        model.eval()
        fold_preds = []
        batch_scores_list = []

        with torch.no_grad():
            for feats, adjs, labels in test_loader:
                feats, adjs = feats.to(DEVICE), adjs.to(DEVICE)
                preds, _, scores = model(feats, adjs)

                fold_preds.append(preds.cpu().numpy())
                if compute_importance:
                    batch_scores_list.append(scores.cpu().numpy())

        y_pred_all[test_idx] = np.concatenate(fold_preds)
        y_true_all[test_idx] = Y_test
        if compute_importance and len(batch_scores_list) > 0:
            all_test_scores = np.concatenate(batch_scores_list, axis=0)  # (N_test, 68)
            avg_scores_fold = np.mean(all_test_scores, axis=0)  # (68,)
            fold_importances.append(avg_scores_fold)

    if verbose:
        print()
    if compute_importance and fold_importances:
        final_importance = np.mean(np.array(fold_importances), axis=0)
    else:
        final_importance = np.zeros(ROI_NUM)

    return y_pred_all, y_true_all, final_importance


def main():
    print(f"Loading table from {TABLE_PATH }...")
    df_table = pd.read_excel(TABLE_PATH, sheet_name="All_trans")
    folder_id_col = df_table.columns[2]
    target_cols = df_table.columns[3:]

    summary_results = []
    df_importance = pd.DataFrame({"ROI_Index": range(1, ROI_NUM + 1)})

    for target in target_cols:
        print(f"\nProcessing Target (CCC+Reg): {target }")
        df_valid = df_table.dropna(subset=[folder_id_col, target])

        X_feats_list, X_adjs_list, Y_labels_list, IDs = [], [], [], []

        for idx, row in df_valid.iterrows():
            fid = str(row[folder_id_col]).strip()
            if fid.endswith(".0"):
                fid = fid[:-2]

            feat, sc = load_data_tensor(fid)
            if feat is not None:
                X_feats_list.append(feat)
                X_adjs_list.append(sc)
                Y_labels_list.append(row[target])
                IDs.append(fid)

        if len(Y_labels_list) < 20:
            print(f"Skipping {target }: Not enough samples")
            continue

        X_feats = np.array(X_feats_list, dtype=np.float32)
        X_adjs = np.array(X_adjs_list, dtype=np.float32)
        Y_labels = np.array(Y_labels_list, dtype=np.float32)
        y_pred_real, y_true_real, roi_scores = run_cross_validation(
            X_feats,
            X_adjs,
            Y_labels,
            IDs,
            epochs=EPOCHS_REAL,
            n_folds=N_FOLDS_REAL,
            batch_size=BATCH_SIZE_REAL,
            compute_importance=True,
            early_stop=False,
            verbose=True,
        )
        df_importance[target] = roi_scores

        real_r, real_p = pearsonr(y_true_real, y_pred_real)
        print(f"True Performance: r={real_r :.4f}, p={real_p :.4e}")
        p_perm_val = float("nan")
        if N_PERMUTATIONS > 0:
            print(f"Running REAL Permutation Test ({N_PERMUTATIONS } times)...")
            perm_rs = []
            for i in range(N_PERMUTATIONS):
                Y_shuffled = np.random.permutation(Y_labels)
                y_pred_perm, _, _ = run_cross_validation(
                    X_feats,
                    X_adjs,
                    Y_shuffled,
                    IDs,
                    epochs=EPOCHS_PERM,
                    n_folds=N_FOLDS_PERM,
                    batch_size=BATCH_SIZE_PERM,
                    compute_importance=False,
                    early_stop=True,
                    early_stop_patience=EARLY_STOP_PATIENCE_PERM,
                    early_stop_min_delta=EARLY_STOP_MIN_DELTA_PERM,
                    verbose=False,
                )
                r_perm, _ = pearsonr(Y_shuffled, y_pred_perm)
                perm_rs.append(r_perm)
                print(f"  Perm {i +1 }/{N_PERMUTATIONS }: r={r_perm :.4f}", end="\r")

            perm_rs = np.array(perm_rs)
            if real_r > 0:
                p_perm_val = (np.sum(perm_rs >= real_r) + 1) / (N_PERMUTATIONS + 1)
            else:
                p_perm_val = (np.sum(perm_rs <= real_r) + 1) / (N_PERMUTATIONS + 1)
            print(f"\nPermutation Result: p_perm={p_perm_val :.4f}")
        safe_target = sanitize_filename(target)
        pd.DataFrame(
            {"FolderID": IDs, "True": y_true_real, "Pred": y_pred_real}
        ).to_csv(os.path.join(OUTPUT_DIR, f"Predict_{safe_target }.csv"), index=False)

        summary_results.append(
            {
                "Metric": target,
                "N": len(Y_labels),
                "Pearson_r": real_r,
                "P_param": real_p,
                "P_perm": p_perm_val,
            }
        )

    if summary_results:
        pd.DataFrame(summary_results).to_excel(
            os.path.join(OUTPUT_DIR, "BrainGNN_Orign_Summary.xlsx"), index=False
        )
        imp_path = os.path.join(OUTPUT_DIR, "ROI_orign_Importance_Rank.xlsx")
        df_importance.to_excel(imp_path, index=False)
        print(f"\nSaved ROI Importance to {imp_path }")


if __name__ == "__main__":
    main()
