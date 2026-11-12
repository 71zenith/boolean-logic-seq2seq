import argparse
import ast
import itertools
import json
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd

QUIT_COMMANDS = {"q", "quit", "exit"}
FIRST_PROMPT = "Expression (e.g. ~(B | D) | (B | A) & (C | B)); 'q' to quit> "
PROMPT = "> "


def parse_args():
    parser = argparse.ArgumentParser(description="Compare a model's truth-table predictions to ground truth.")
    parser.add_argument("--model", "-m", type=Path, required=True, help="Path to the .keras model file")
    parser.add_argument("--preprocessing", "-p", type=Path, required=True, help="Path to the preprocessing JSON file")
    return parser.parse_args()


def load_artifacts(model_path, preprocessing_path):
    for path in (model_path, preprocessing_path):
        if not path.is_file():
            raise FileNotFoundError(f"Can't find {path.resolve()}")

    with preprocessing_path.open("r", encoding="utf-8") as f:
        pre = json.load(f)

    lookup = {str(k): int(v) for k, v in pre["lookup_table"].items()}
    max_exprs_len = int(pre["max_exprs_len"])
    max_truth_len = int(pre["max_truth_len"])

    start = time.perf_counter()
    model = keras.saving.load_model(model_path, compile=False)
    load_time = time.perf_counter() - start

    print(f"Loaded {model_path} using Keras {keras.__version__} / {keras.backend.backend()} backend")
    print(f"Model load time: {load_time:.2f}s")
    print(f"Input max length: {max_exprs_len}; output width: {max_truth_len}")
    return model, lookup, max_exprs_len, max_truth_len


def encode(text, lookup, max_len):
    """Turn a canonical expression string into a padded (1, max_len) int array."""
    unknown = set(text) - set(lookup)
    if unknown:
        raise ValueError(f"Characters absent from training vocabulary: {sorted(unknown)}")
    if len(text) > max_len:
        raise ValueError(f"Model input has {len(text)} characters; model limit is {max_len}.")
    x = np.zeros((1, max_len), dtype=np.int32)
    x[0, : len(text)] = [lookup[c] for c in text]
    return x


def canonical(node, names):
    """Rewrite the AST into the model's format: A-E variables, sorted operands."""
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Invert, ast.Not)):
        return "~" + canonical(node.operand, names)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        op = "&" if isinstance(node.op, ast.BitAnd) else "|"
        a, b = sorted((canonical(node.left, names), canonical(node.right, names)))
        return f"({a} {op} {b})"
    raise ValueError("Use variables with &, |, ~, and parentheses.")


def evaluate(node, env):
    """Ground-truth evaluation of the (already validated) AST."""
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        return not evaluate(node.operand, env)
    a, b = evaluate(node.left, env), evaluate(node.right, env)
    return (a and b) if isinstance(node.op, ast.BitAnd) else (a or b)


def compare_truth_table(expr, model, lookup, max_exprs_len, max_truth_len):
    tree = ast.parse(expr.strip(), mode="eval").body
    variables = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})

    if not 1 <= len(variables) <= 5:
        raise ValueError("Expression must contain 1-5 variables.")

    rows = list(itertools.product([False, True], repeat=len(variables)))
    if len(rows) > max_truth_len:
        raise ValueError(f"Truth table has {len(rows)} rows; model output width is {max_truth_len}.")

    model_expr = canonical(tree, dict(zip(variables, "ABCDE")))  # also validates the AST
    actual = [evaluate(tree, dict(zip(variables, bits))) for bits in rows]

    x = encode(model_expr, lookup, max_exprs_len)
    probs = np.asarray(model.predict(x, verbose=0))[0][: len(rows)]
    predicted = probs >= 0.5

    result = pd.DataFrame(rows, columns=variables)
    result["Actual"] = actual
    result["Predicted"] = predicted
    result["Probability"] = probs
    result["Correct"] = result["Actual"] == result["Predicted"]

    print(f"Model input: {model_expr}")
    wrong = int((~result["Correct"]).sum())
    print("Exact match" if wrong == 0 else f"{wrong} of {len(rows)} rows wrong")
    return result


def main():
    args = parse_args()

    try:
        import readline
    except ImportError:
        pass

    try:
        model, lookup, max_exprs_len, max_truth_len = load_artifacts(args.model, args.preprocessing)
    except FileNotFoundError as exc:
        raise SystemExit(f"Error: {exc}")

    prompt = FIRST_PROMPT
    while True:
        try:
            expression = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        prompt = PROMPT

        if not expression:
            continue
        if expression.lower() in QUIT_COMMANDS:
            break

        try:
            result = compare_truth_table(expression, model, lookup, max_exprs_len, max_truth_len)
        except SyntaxError:
            print("Error: could not parse that expression.\n")
            continue
        except (ValueError, KeyError, RecursionError) as exc:
            print(f"Error: {exc}\n")
            continue

        print(result.to_string(index=False))
        print(f"\nExact truth table match: {bool(result['Correct'].all())}\n")


if __name__ == "__main__":
    main()
