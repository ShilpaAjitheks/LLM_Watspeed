import os
import gc
import copy
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CoSENTLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.preprocessing import MinMaxScaler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def calculate_gap(results: list) -> float:
    close_sims = [r["cosine_similarity"] for r in results if r["expected_association"] == "close"]
    far_sims   = [r["cosine_similarity"] for r in results if r["expected_association"] == "far"]
    close_mean, far_mean = np.mean(close_sims), np.mean(far_sims)
    gap = close_mean - far_mean

    print(f"Close — mean: {close_mean:.3f}  (n={len(close_sims)})")
    print(f"Far   — mean: {far_mean:.3f}  (n={len(far_sims)})")
    print(f"Gap:   {gap:.3f}", end="  →  ")

    if   gap >= 0.2: print("good signal")
    elif gap >= 0.1: print("borderline")
    else:            print("too weak — switch models")
    return gap


def evaluate_pairs_st(word_pairs: list, model: SentenceTransformer, label: str = "") -> list:
    results = []
    for word_a, word_b, expected in word_pairs:
        sim = cosine_similarity(
            model.encode(word_a).tolist(),
            model.encode(word_b).tolist(),
        )
        results.append({
            "word_a": word_a,
            "word_b": word_b,
            "expected_association": expected,
            "model": label,
            "cosine_similarity": round(sim, 4),
        })
    if label:
        print(f"\nModel: {label}")
    print(pd.DataFrame(results).drop(columns="model").to_string(index=False))
    return results


# ---------------------------------------------------------------------------
# Step 1 — Load domain pairs
# ---------------------------------------------------------------------------

from domain_pairs import get_train_examples

all_pairs = get_train_examples()
print(f"Total pairs: {len(all_pairs)}  (target ≥ 50)")

# ---------------------------------------------------------------------------
# Step 2 — Baseline gap
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "all-mpnet-base-v2"

base_model = SentenceTransformer(BASE_MODEL_ID)
print(f"Loaded '{BASE_MODEL_ID}'  —  dim={base_model.get_embedding_dimension()}")

print("\n=== Baseline ===")
baseline_results = evaluate_pairs_st(all_pairs[70:], base_model, label=BASE_MODEL_ID)
baseline_gap = calculate_gap(baseline_results)

# ---------------------------------------------------------------------------
# Step 3 — Build Hugging Face Dataset
# ---------------------------------------------------------------------------

label_map = {"close": 1.0, "far": 0.0}

hf_data = Dataset.from_dict({
    "sentence1": [p[0] for p in all_pairs],
    "sentence2": [p[1] for p in all_pairs],
    "label":     [label_map[p[2]] for p in all_pairs],
})

split    = hf_data.train_test_split(test_size=0.15, seed=0)
train_ds = split["train"]
eval_ds  = split["test"]

print(hf_data)
print(f"Train: {len(train_ds)}  |  Eval: {len(eval_ds)}")

# ---------------------------------------------------------------------------
# Step 4 — Fine-tune with CoSENTLoss
# ---------------------------------------------------------------------------

ft_model = copy.deepcopy(base_model)
loss = CoSENTLoss(ft_model)

evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_ds["sentence1"],
    sentences2=eval_ds["sentence2"],
    scores=eval_ds["label"],
    name="domain-eval",
)

args = SentenceTransformerTrainingArguments(
    output_dir="models/ft-domain-embedding",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    warmup_steps=5,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_domain-eval_spearman_cosine",
    dataloader_pin_memory=False,
    logging_steps=1,
    report_to="none",
)

trainer = SentenceTransformerTrainer(
    model=ft_model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    loss=loss,
    evaluator=evaluator,
)

gc.collect()
torch.cuda.empty_cache()
trainer.train()

# ---------------------------------------------------------------------------
# Step 5 — Compare baseline vs fine-tuned
# ---------------------------------------------------------------------------

print("\n=== Fine-tuned ===")
ft_results = evaluate_pairs_st(all_pairs, ft_model, label=f"{BASE_MODEL_ID}-finetuned")
ft_gap     = calculate_gap(ft_results)

improvement = ft_gap - baseline_gap
print(f"\nBaseline gap:   {baseline_gap:.3f}")
print(f"Fine-tuned gap: {ft_gap:.3f}")
print(f"Improvement:    {improvement:+.3f}", end="  →  ")

if   improvement > 0.05: print("fine-tuning helped — use the fine-tuned model")
elif improvement > 0:    print("marginal — try more domain pairs or more epochs")
else:                    print("no gain — add more pairs or try a larger base model")

# Per-pair delta
df_base = pd.DataFrame(baseline_results).rename(columns={"cosine_similarity": "base_sim"})
df_ft   = pd.DataFrame(ft_results).rename(columns={"cosine_similarity": "ft_sim"})

comparison = df_base[["word_a", "word_b", "expected_association", "base_sim"]].copy()
comparison["ft_sim"] = df_ft["ft_sim"]
comparison["delta"]  = (comparison["ft_sim"] - comparison["base_sim"]).round(4)

print(comparison.sort_values("delta", ascending=False).reset_index(drop=True).to_string(index=False))

# ---------------------------------------------------------------------------
# Step 6 — Save the fine-tuned model
# ---------------------------------------------------------------------------

save_path = "models/ft-domain-embedding/cuisine_mpnet_ft"
os.makedirs(save_path, exist_ok=True)
ft_model.save(save_path)
print(f"\nSaved → {save_path}")

# ---------------------------------------------------------------------------
# Step 7 — Save results to CSV and normalise similarity scores
# ---------------------------------------------------------------------------

os.makedirs("data", exist_ok=True)
df = pd.DataFrame(ft_results).drop(columns="model")
df.to_csv("data/embedding_eval.csv", index=False)
print(f"Saved {len(df)} pairs → data/embedding_eval.csv")

# Normalise similarity scores to [0, 1]
cosine_data = df["cosine_similarity"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
df["normalized_similarity"] = scaler.fit_transform(cosine_data)

overall_min = df["cosine_similarity"].min()
overall_max = df["cosine_similarity"].max()
print(f"\nNormalised from [{overall_min:.4f}, {overall_max:.4f}] → [0, 1]")

for label in ("close", "far"):
    subset = df[df["expected_association"] == label]["normalized_similarity"]
    print(f"Normalised '{label}' range: [{subset.min():.4f}, {subset.max():.4f}]")
