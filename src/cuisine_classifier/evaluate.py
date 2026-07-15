#!/usr/bin/env python3
"""
Evaluation script for cuisine classifier.
Runs ablation study: baseline (dish name only) → context-enhanced → retrieval-augmented.
"""

import sys
import os
import pandas as pd
from sklearn.metrics import f1_score
from cuisine_classifier.__main__ import load_config, load_dataset, predict_cuisine


def evaluate_classifier(test_dishes, config, leg_label="", dataset=None, use_retrieval=False, vector_store=None, model_name=None, use_rag=False, rag_vector_store=None):
    """
    Evaluate classifier on test dishes.

    Args:
        test_dishes: List of (dish_name, expected_cuisine) tuples
        config: Configuration dictionary
        leg_label: Display label for progress output (e.g. "No Context")
        dataset: Optional recipe dataset for context lookup (None = dish name only)
        use_retrieval: Whether to use ChromaDB few-shot retrieval
        vector_store: VectorStore instance (required if use_retrieval=True)

    Returns:
        tuple: (accuracy, correct_count, total_count, predictions)
    """
    predictions = []
    correct = 0
    total = len(test_dishes)

    for i, (dish_name, expected) in enumerate(test_dishes, 1):
        print(f"  [{leg_label}] {i}/{total} — {dish_name[:50]}", flush=True)
        output, _, __, ___, ____ = predict_cuisine(
            dish_name, config,
            dataset=dataset,
            use_retrieval=use_retrieval,
            vector_store=vector_store,
            model_name=model_name,
            use_rag=use_rag,
            rag_vector_store=rag_vector_store,
        )
        predicted = output["Cuisine"] if output else "ERROR"
        is_correct = predicted.lower() == expected.lower()
        if is_correct:
            correct += 1
        predictions.append({
            "dish": dish_name,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = (correct / total * 100) if total > 0 else 0
    y_true = [p["expected"] for p in predictions]
    y_pred = [p["predicted"] for p in predictions]
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    return accuracy, correct, total, predictions, macro_f1


def print_results(mode, accuracy, correct, total, predictions, macro_f1, verbose=False):
    """Print evaluation results."""
    print(f"\n{'='*65}")
    print(f"{mode.upper()} MODE")
    print(f"{'='*65}")
    print(f"Accuracy:   {accuracy:.1f}% ({correct}/{total})")
    print(f"Macro F1:   {macro_f1:.1f}%")

    if verbose:
        print(f"\nDetailed Results:")
        for pred in predictions:
            status = "OK" if pred["correct"] else "XX"
            print(
                f"  [{status}] {pred['dish'][:40]:40} "
                f"| GT: {pred['expected']:8} "
                f"| Pred: {pred['predicted']}"
            )
    else:
        failures = [p for p in predictions if not p["correct"]]
        if failures:
            print(f"\nFailures:")
            for pred in failures:
                print(
                    f"  [XX] {pred['dish'][:40]:40} "
                    f"| GT: {pred['expected']:8} "
                    f"| Pred: {pred['predicted']}"
                )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate cuisine classifier with ablation study")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name to use (overrides config)")
    parser.add_argument("--eval-set", type=str, default=None, help="Path to eval CSV (Name, expected columns)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed per-dish predictions (all results)")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline leg and run context-enhanced only")
    parser.add_argument("--retrieval", action="store_true", help="Run retrieval-augmented mode as third ablation leg")
    parser.add_argument("--embed-model", type=str, default=None, help="Embedding model for retrieval (required with --retrieval)")
    parser.add_argument("--rag", action="store_true", help="Run Week 6 cuisine knowledge RAG as an ablation leg")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    package_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(package_dir))

    eval_path = args.eval_set or os.path.join(project_root, "data", "prompt_eval_cuisine.csv")
    if not os.path.exists(eval_path):
        print(f"Error: Eval set not found at {eval_path}", file=sys.stderr)
        sys.exit(1)

    if eval_path.endswith(".xlsx"):
        eval_df = pd.read_excel(eval_path)
    else:
        eval_df = pd.read_csv(eval_path)

    if "expert_label" in eval_df.columns and "expected" not in eval_df.columns:
        eval_df = eval_df.rename(columns={"expert_label": "expected"})

    if not {"Name", "expected"}.issubset(eval_df.columns):
        print("Error: Eval file must have 'Name' and 'expected' (or 'expert_label') columns", file=sys.stderr)
        sys.exit(1)

    dataset_path = os.path.join(project_root, config["dataset"]["path"])
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    if dataset is not None:
        print(f"Loaded {len(dataset)} recipes")

    test_dishes = [(row["Name"], row["expected"].strip()) for _, row in eval_df.iterrows()]
    active_model = args.model or config["model"]["name"]
    print(f"\nRunning ablation study with {len(test_dishes)} test dishes")
    print(f"Using model: {active_model}")

    # Leg 1 — baseline: dish name only (no dataset context, no retrieval)
    base_accuracy = base_correct = base_total = base_predictions = None
    if not args.skip_baseline:
        print("\n[1/3] Baseline — dish name only (no context, no retrieval)")
        print(f"{'-'*65}")
        base_accuracy, base_correct, base_total, base_predictions, base_f1 = evaluate_classifier(
            test_dishes, config, leg_label="No Context", dataset=None, use_retrieval=False, model_name=active_model,
        )

    # Leg 2 — context-enhanced: dataset lookup for ingredients + description
    default_model = config["model"]["name"]
    is_adapter = active_model != default_model
    ctx_leg_label = f"Context-Enhanced [{active_model}]" if is_adapter else "Context-Enhanced — dataset lookup (ingredients + description)"
    print(f"\n[2/3] {ctx_leg_label}")
    print(f"{'-'*65}")
    ctx_accuracy, ctx_correct, ctx_total, ctx_predictions, ctx_f1 = evaluate_classifier(
        test_dishes, config, leg_label="With Context", dataset=dataset, use_retrieval=False, model_name=active_model,
    )

    # Leg 3 — retrieval-augmented: context + ChromaDB few-shot (only if --retrieval passed)
    rag_accuracy = rag_correct = rag_total = rag_predictions = None
    if args.retrieval:
        if not args.embed_model:
            print("Error: --embed-model is required when using --retrieval", file=sys.stderr)
            print("  Options: nomic-embed-text  |  all-mpnet-base-v2", file=sys.stderr)
            sys.exit(1)
        try:
            sys.path.insert(0, project_root)
            from retrieval.vector_store import VectorStore
            vector_store = VectorStore(model=args.embed_model)
            if vector_store.count() == 0:
                print(f"Error: Vector store for '{args.embed_model}' is empty.", file=sys.stderr)
                print(f"  Run: uv run python retrieval/populate.py --model {args.embed_model}", file=sys.stderr)
                sys.exit(1)
            print(f"\n[3/3] Retrieval-Augmented — context + ChromaDB few-shot ({args.embed_model})")
            print(f"{'-'*65}")
            rag_accuracy, rag_correct, rag_total, rag_predictions, rag_f1 = evaluate_classifier(
                test_dishes, config, leg_label="With Retrieval", dataset=dataset,
                use_retrieval=True, vector_store=vector_store, model_name=active_model,
            )
        except Exception as e:
            print(f"Error loading vector store: {e}", file=sys.stderr)
            sys.exit(1)

    # Leg 4 — Week 6 cuisine knowledge RAG (only if --rag passed)
    knowledge_rag_accuracy = knowledge_rag_correct = knowledge_rag_total = knowledge_rag_predictions = None
    if args.rag:
        try:
            sys.path.insert(0, project_root)
            from retrieval.vector_store import VectorStore
            from retrieval.knowledge_store import get_knowledge_collection
            rag_vs = VectorStore(model="mpnet")
            if rag_vs.count() == 0:
                print("Error: mpnet vector store is empty — run: uv run python retrieval/populate.py --model mpnet", file=sys.stderr)
                sys.exit(1)
            kc = get_knowledge_collection()
            if kc.count() < 6:
                print("Error: knowledge store not populated — run: uv run python retrieval/knowledge_store.py", file=sys.stderr)
                sys.exit(1)
            leg_num = "4" if args.retrieval else "3"
            print(f"\n[{leg_num}/3] Cuisine Knowledge RAG — ambiguity signal + retrieved cuisine cards")
            print(f"{'-'*65}")
            knowledge_rag_accuracy, knowledge_rag_correct, knowledge_rag_total, knowledge_rag_predictions, knowledge_rag_f1 = evaluate_classifier(
                test_dishes, config, leg_label="Knowledge RAG", dataset=dataset,
                model_name=active_model, use_rag=True, rag_vector_store=rag_vs,
            )
        except Exception as e:
            print(f"Error initialising RAG components: {e}", file=sys.stderr)
            sys.exit(1)

    # Print results
    if base_predictions is not None:
        print_results("Baseline (dish name only)", base_accuracy, base_correct, base_total, base_predictions, base_f1, args.verbose)
    print_results(ctx_leg_label, ctx_accuracy, ctx_correct, ctx_total, ctx_predictions, ctx_f1, args.verbose)

    if rag_predictions is not None:
        print_results(f"Retrieval-Augmented ({args.embed_model})", rag_accuracy, rag_correct, rag_total, rag_predictions, rag_f1, args.verbose)

    if knowledge_rag_predictions is not None:
        print_results("Cuisine Knowledge RAG (Week 6)", knowledge_rag_accuracy, knowledge_rag_correct, knowledge_rag_total, knowledge_rag_predictions, knowledge_rag_f1, args.verbose)

    # Comparison
    print(f"\n{'='*65}")
    print(f"COMPARISON")
    print(f"{'='*65}")
    if base_predictions is not None:
        print(f"Baseline (dish name only):        {base_accuracy:.1f}%  (Macro F1: {base_f1:.1f}%)")
    print(f"{ctx_leg_label}: {ctx_accuracy:.1f}%  (Macro F1: {ctx_f1:.1f}%)")
    if base_predictions is not None:
        ctx_improvement = ctx_accuracy - base_accuracy
        print(f"Context improvement:              {ctx_improvement:+.1f} percentage points")
        if ctx_improvement > 0:
            print(f"\n[OK] Dataset context improved accuracy!")
        elif ctx_improvement < 0:
            print(f"\n[XX] Dataset context decreased accuracy")
        else:
            print(f"\n[--] No change from adding context")

    if rag_predictions is not None:
        rag_improvement = rag_accuracy - ctx_accuracy
        print(f"Retrieval-Augmented:              {rag_accuracy:.1f}%  (Macro F1: {rag_f1:.1f}%)")
        print(f"Retrieval improvement over ctx:   {rag_improvement:+.1f} percentage points")

    if knowledge_rag_predictions is not None:
        kr_improvement = knowledge_rag_accuracy - ctx_accuracy
        print(f"Cuisine Knowledge RAG:            {knowledge_rag_accuracy:.1f}%  (Macro F1: {knowledge_rag_f1:.1f}%)")
        print(f"RAG improvement over ctx:         {kr_improvement:+.1f} percentage points")

    # Per-category breakdown
    print(f"\nPer-category breakdown (context-enhanced):")
    df = pd.DataFrame(ctx_predictions)
    for category in sorted(df["expected"].unique()):
        subset = df[df["expected"] == category]
        cat_correct = subset["correct"].sum()
        cat_total = len(subset)
        print(f"  {category:10} {cat_correct}/{cat_total}")

    print(f"\nTest dishes: {len(test_dishes)}")


if __name__ == "__main__":
    main()
