"""Start here.  The reasoner API, level by level:   python api.py

The system answers a question in three interactions (see note.md):

  1. entity index   which entities exist -- constrains what the LLM may generate
  2. properties(e)  which relations and rules can answer for that entity
  3. get/count/ask  run the rule and return the results

A property string carries the query direction: `own` binds the entity as the
subject (OWN(e, ?) -- what e owns), `is_own` binds it as the object
(OWN(?, e) -- who owns e).
"""
import bisect
import time
from collections import defaultdict
import pandas as pd
from reasoner import load

ENTITIES, QA = "data/entities.csv", "data/qa_answers.csv"

CALL = {
    "ask":     'R.ask(x, p, y)',
    "count":   'R.count(y, "is_" + p)',
    "list":    'R.get(y, "is_" + p)',
    "more":    'R.count(x, p) > R.count(z, p)',
    "and_not": 'R.ask(x, p, y) and not R.ask(z, p, y)',
    "top":     'max([x, y, z], key=lambda e: R.count(e, p))',
}

reasoner_db = load()

def entity_index():
    """Level 1: the sorted entity names, i.e. what the prefix tree indexes."""
    _index = sorted(pd.read_csv(ENTITIES, keep_default_na=False).entity)
    return _index


def property_strings(entity):
    """Level 2: the property strings that answer for `entity`."""
    return sorted(reasoner_db.properties(entity))

def get(entity, prop):
    """Level 3: the entities that answer `prop` for `entity`."""
    return reasoner_db.get(entity, prop)

def count(entity, prop):
    """Level 3: how many entities answer `prop` for `entity`."""
    return reasoner_db.count(entity, prop)

def ask(subject, prop, obj):
    """Level 3: whether one single fact holds."""
    return reasoner_db.ask(subject, prop, obj)


def show(result):
    """Format a result the way the benchmark stores it."""
    if isinstance(result, bool):
        return "yes" if result else "no"
    if isinstance(result, list):
        return "; ".join(result)
    return str(result)


def banner(n, title):
    print("\n" + "=" * 78 + "\n%d. %s\n" % (n, title) + "=" * 78)


if __name__ == "__main__":
    qa = pd.read_csv(QA, keep_default_na=False)

    ## level 1: which entities exist
    banner(1, "entity index -- which entities exist")
    index = entity_index()
    print(len(index), "entities:", index[:2], "...", index[-2:])


    ## level 2: what can answer for that entity
    sample_entity = index[len(index) // 2]
    banner(2, "properties(entity) -- what can answer for it")
    print("entity:", sample_entity)
    print("Properties", property_strings(sample_entity))
    print("Details:")
    for name in property_strings(sample_entity):
        print("   %-22s %7d results" % (name, count(sample_entity, name)))

    banner(2, "Example of APIs")
    ## companies controlled by that entity
    controlled = get(sample_entity, "control")
    print("<Entity, 'control', ?>", controlled[:3])
    print("\n ## \n")
    is_own = get(sample_entity, "is_own")
    print("<?, 'own', Entity>", is_own[:3])


    ## example with QAs
    type_of_questions = qa.kind.value_counts()
    banner(3, "QA examples -- what the system can answer")
    for kind, _ in type_of_questions.items():
        print("%-8s question" % (kind))
        question_one = qa[qa.kind == kind].iloc[0]

        type_question = question_one["kind"]
        kind_question = question_one["kind"]
        property_question = question_one["property"]
        x_question = question_one["x"]
        y_question = question_one["y"]
        z_question = question_one["z"]
        answer_question = question_one["answer"]
        print("\nExample question:", question_one["question"])
        print("Approach:")

        if kind_question == "ask":
            ## these are yes and no questions, so we have to verify the fact in the DB.
            print("Query:", f"ask({x_question}, {property_question}, {y_question})")
            print("Answer:", ask(x_question, property_question, y_question))
            print("Expected answer:", answer_question)

        if kind_question == "count":
            print("Query:", f"count({y_question}, '{property_question}')")
            print("Answer:", count(y_question, property_question))
            print("Expected answer:", answer_question)

        if kind_question == "list":
            print("Query:", f"get({y_question}, '{property_question}')")
            print("Answer:", get(y_question, property_question))
            print("Expected answer:", answer_question)

        if kind_question == "more":
            print("Query:", f"count({x_question}, {property_question}) > count({z_question}, {property_question})")
            print("Answer part 1:", count(x_question, property_question))
            print("Answer part 2:", count(z_question, property_question))
            print("Answer:", count(x_question, property_question) > count(z_question, property_question))
            print("Expected answer:", answer_question)

        if kind_question == "and_not":
            print("Query:", f"ask({x_question}, {property_question}, {y_question}) and not ask({z_question}, {property_question}, {y_question})")
            print("Answer part 1:", ask(x_question, property_question, y_question))
            print("Answer part 2:", ask(z_question, property_question, y_question))
            print("Answer:", ask(x_question, property_question, y_question) and not ask(z_question, property_question, y_question))
            print("Expected answer:", answer_question)

        if kind_question == "top":
            print("Query:", f"max([{x_question}, {y_question}, {z_question}], key=lambda e: count(e, {property_question}))")
            print("Answer part 1:", count(x_question, property_question))
            print("Answer part 2:", count(y_question, property_question))
            print("Answer part 3:", count(z_question, property_question))
            print("Answer:", max([x_question, y_question, z_question], key=lambda e: count(e, property_question)))
            print("Expected answer:", answer_question)
