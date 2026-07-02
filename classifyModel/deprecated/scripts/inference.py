import numpy as np
import pandas as pd
import torch

def predict_from_probs(
    probs,
    use_conifer_threshold=False,
    conifer_threshold=0.5,
    conifer_class_idx=0,
    broadleaf_class_idx=1,
):
    if use_conifer_threshold:
        conifer_prob = probs[:, conifer_class_idx]
        preds = np.where(
            conifer_prob >= conifer_threshold,
            conifer_class_idx,
            broadleaf_class_idx,
        )
        return preds.astype(np.int64)

    return np.argmax(probs, axis=1).astype(np.int64)


@torch.no_grad()
def evaluate_with_probs(
    model,
    dataloader,
    criterion,
    device,
    use_conifer_threshold=False,
    conifer_threshold=0.5,
):
    model.eval()

    running_loss = 0.0
    n_total = 0
    n_correct = 0

    all_true = []
    all_pred = []
    all_probs = []
    all_unique_ids = []
    all_years = []
    all_genus_names = []
    all_valid_measurements = []

    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        annual_metrics = batch["annual_metrics"].to(device, non_blocking=True)
        y = batch["y_broad"].to(device, non_blocking=True)

        logits, _ = model(x, lengths, annual_metrics)
        probs = torch.softmax(logits, dim=1)
        loss = criterion(logits, y)

        probs_np = probs.cpu().numpy()
        preds_np = predict_from_probs(
            probs_np,
            use_conifer_threshold=use_conifer_threshold,
            conifer_threshold=conifer_threshold,
        )

        preds = torch.tensor(preds_np, device=device, dtype=torch.long)

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        n_total += batch_size
        n_correct += (preds == y).sum().item()

        all_true.append(y.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        all_unique_ids.extend(batch["unique_ids"])
        all_years.extend(batch["years"])
        all_genus_names.extend(batch["genus_names"])
        all_valid_measurements.extend(batch["valid_measurements"].cpu().numpy().tolist())

    return {
        "loss": running_loss / n_total,
        "acc": n_correct / n_total,
        "y_true": np.concatenate(all_true),
        "y_pred": np.concatenate(all_pred),
        "probs": np.concatenate(all_probs, axis=0),
        "unique_ids": all_unique_ids,
        "years": all_years,
        "genus_names": all_genus_names,
        "valid_measurements": all_valid_measurements,
    }

def fuse_tree_year_probs_simple(
    unique_ids,
    years,
    y_true,
    probs,
    weights,
    class_names,
):
    df = pd.DataFrame({
        "unique_id": unique_ids,
        "year": years,
        "y_true": y_true,
        "weight": weights,
    })

    for i in range(probs.shape[1]):
        df[f"p_{i}"] = probs[:, i]

    tree_rows = []

    for uid, g in df.groupby("unique_id"):
        p = g[[f"p_{i}" for i in range(probs.shape[1])]].values
        w = g["weight"].values.astype(float)

        # Avoid divide-by-zero
        if w.sum() == 0:
            w = np.ones_like(w)

        w = w / w.sum()

        # 🔥 Core logic: weighted log-probability fusion
        logp = np.log(np.clip(p, 1e-8, 1.0))
        fused_logp = (w[:, None] * logp).sum(axis=0)

        # Convert back to probabilities (optional but clean)
        fused_prob = np.exp(fused_logp)
        fused_prob = fused_prob / fused_prob.sum()

        pred = int(np.argmax(fused_prob))

        tree_rows.append({
            "unique_id": uid,
            "y_true": int(g["y_true"].iloc[0]),
            "y_pred_fused": pred,
            "prob_conifer": fused_prob[0],
            "total_weight": float(g["weight"].sum()),
        })

    tree_df = pd.DataFrame(tree_rows)

    acc = (tree_df["y_true"] == tree_df["y_pred_fused"]).mean()

    return tree_df, acc

def fuse_tree_year_probs_temporal(
    unique_ids,
    years,
    y_true,
    probs,
    weights,
    class_names,
    conifer_class_idx=0,
    conifer_log_bias=0.05,
    conifer_max_weight=0.0,
):
    df = pd.DataFrame({
        "unique_id": unique_ids,
        "year": years,
        "y_true": y_true,
        "weight": weights,
    })

    probs = np.asarray(probs, dtype=np.float64)

    for i in range(probs.shape[1]):
        df[f"p_{i}"] = probs[:, i]

    rows = []
    prob_cols = [f"p_{i}" for i in range(probs.shape[1])]

    for uid, g in df.groupby("unique_id", sort=False):
        p = g[prob_cols].to_numpy(dtype=np.float64)
        w = g["weight"].to_numpy(dtype=np.float64)

        if w.sum() <= 0:
            w = np.ones_like(w)

        w = w / w.sum()

        logp = np.log(np.clip(p, 1e-8, 1.0))

        # Original stable coverage-weighted log fusion
        fused_logp = (w[:, None] * logp).sum(axis=0)

        # Conifer-only max-year boost
        max_conifer_logp = logp[:, conifer_class_idx].max()
        fused_logp[conifer_class_idx] = (
            (1.0 - conifer_max_weight) * fused_logp[conifer_class_idx]
            + conifer_max_weight * max_conifer_logp
        )

        # Small conifer bias
        fused_logp[conifer_class_idx] += conifer_log_bias

        fused_prob = np.exp(fused_logp - np.max(fused_logp))
        fused_prob = fused_prob / fused_prob.sum()

        pred = int(np.argmax(fused_logp))
        true = int(g["y_true"].iloc[0])

        rows.append({
            "unique_id": uid,
            "n_years": int(len(g)),
            "total_weight": float(g["weight"].sum()),
            "y_true": true,
            "y_pred_fused": pred,
            "y_true_name": class_names[true],
            "y_pred_fused_name": class_names[pred],
            "prob_conifer": float(fused_prob[conifer_class_idx]),
        })

    tree_df = pd.DataFrame(rows)
    acc = (tree_df["y_true"] == tree_df["y_pred_fused"]).mean()

    return tree_df, acc
