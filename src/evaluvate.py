# src/evaluate.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from rag_pipeline import answer
import re
def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    # strip punctuation
    s = re.sub(r'[^\w\s]', '', s)
    return s

def f1_score(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return int(pred_tokens==gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def run_eval(n_samples=100):
    ds_dict = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    print(ds_dict)
    split_name = "train" if "train" in ds_dict.keys() else next(iter(ds_dict.keys()))
    print("Using split:", split_name)
    ds = ds_dict[split_name]

    total_em=0
    total_f1=0.0
    for i,ex in enumerate(ds):
        if i>=n_samples: break
        q = ex.get("question", "")
        gold = ex.get("answer") or ex.get("answers") or ""
        if isinstance(gold, (list, tuple)):
            gold = gold[0] if gold else ""
        gold = str(gold)

        result = answer(q, top_k=4)
        pred = result.get("answer", "") if isinstance(result, dict) else str(result)
        if isinstance(pred, (list, tuple)):
            pred = pred[0] if pred else ""
        pred = str(pred)

        em = int(normalize_answer(pred) == normalize_answer(gold))
        f1 = f1_score(pred, gold)
        total_em += em
        total_f1 += f1
        print(i, q, "->", pred[:150], "EM", em, "F1", f1)
    print("EM:", total_em/n_samples, "F1:", total_f1/n_samples)

if __name__ == "__main__":
    run_eval(50)
