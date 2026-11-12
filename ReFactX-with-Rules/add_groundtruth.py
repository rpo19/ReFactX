"""Add the ground-truth answers to data/qa.csv, computed with the reasoner."""
import pandas as pd
from reasoner import load


def answer(R, r):
    """Ground truth of one question, dispatched on its kind."""
    p, x, y, z = r["property"], r["x"], r["y"], r["z"]
    yn = lambda b: "yes" if b else "no"
    if r["kind"] == "ask":
        return yn(R.ask(x, p, y))
    if r["kind"] == "count":
        return str(R.count(y,  p))
    if r["kind"] == "list":
        return "; ".join(R.get(y,  p))
    if r["kind"] == "more":
        return yn(R.count(x, p) > R.count(z, p))
    if r["kind"] == "and_not":
        return yn(R.ask(x, p, y) and not R.ask(z, p, y))
    if r["kind"] == "top":
        return max([x, y, z], key=lambda e: R.count(e, p))
    raise ValueError(r["kind"])


if __name__ == "__main__":
    R = load()
    qa = pd.read_csv("data/qa.csv", keep_default_na=False)
    qa["answer"] = [answer(R, r) for _, r in qa.iterrows()]
    qa["n_answers"] = [len(a.split("; ")) if t == "list" else 1
                       for a, t in zip(qa.answer, qa.answer_type)]
    qa.to_csv("data/qa_answers.csv", index=False)
    print(len(qa), "answered ->  data/qa_answers.csv")
    print(qa[qa.answer_type == "yes_no"].answer.value_counts().to_string())
    print("max list size:", qa.n_answers.max())
