"""Tests for utils/generate_backtracking_traces.py."""
import json
import tempfile
import unittest
from pathlib import Path

from utils.generate_backtracking_traces import (
    DEFAULT_PROMPT,
    TASK_PROMPT,
    SourceFactOracle,
    build_backtracking_user_message,
    build_injected_trace,
    build_record,
    inject_fact_result,
    inject_one,
    load_generator_prompt,
    load_system_prompt,
    main,
    parse_fact_calls,
    parse_open_fact_calls,
    read_candidates,
    rejected_from_trace,
    validate_backtracking_trace,
)

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
REPO = PROMPTS.parent

RECORD = {
    "question": "Who directed Slumdog Millionaire?",
    "gt_answer": "Danny Boyle",
    "prediction": ["Danny Boyle"],
    "input_sample": {"complexityType": "generic"},
    "triples": [
        " <Slumdog Millionaire> <description> <2008 film> .",
        " <Slumdog Millionaire> <director> <Danny Boyle> .",
    ],
    "reached_max_tokens": False,
    "evaluation": {"correct": 1, "precision": 1, "recall": 1, "f1": 1, "dont_know": 0},
}

GOOD_TRACE = (
    'I need to answer: Who directed Slumdog Millionaire? The most direct first step seems '
    'to be the relation "description" of "Slumdog Millionaire", so I will check that first.\n'
    "<fact> <Slumdog Millionaire> <description> <2008 film> . </fact>\n"
    'This returns "2008 film". Taken on its own it is a plausible candidate, but the question '
    'asks for the "director" of "Slumdog Millionaire", not the "description" of '
    '"Slumdog Millionaire", so "2008 film" is not the answer. That\'s not the right path: '
    "I reject this candidate and back out of this branch.\n"
    'I will instead retrieve the "director" of "Slumdog Millionaire".\n'
    "<fact> <Slumdog Millionaire> <director> <Danny Boyle> . </fact>\n"
    'The retrieved facts give Danny Boyle, which supports the answer.\n'
    '<answer>\n["Danny Boyle"]\n</answer>'
)


class TestPrompts(unittest.TestCase):
    """The generator prompt wraps the real task prompt and asks for Q+A examples."""

    def test_wrapper_includes_task_prompt(self):
        prompt = load_generator_prompt(REPO / DEFAULT_PROMPT, REPO / TASK_PROMPT)
        self.assertIn("Synthetic training-example generator", prompt)
        self.assertIn("Real task prompt", prompt)
        self.assertIn("Fact-Grounded Question Answering", prompt)  # task prompt body
        self.assertNotIn("{{TASK_PROMPT}}", prompt)

    def test_wrapper_template_has_placeholder(self):
        text = load_system_prompt(REPO / DEFAULT_PROMPT)
        self.assertIn("{{TASK_PROMPT}}", text)

    def test_defaults_point_to_generator_prompt(self):
        self.assertTrue(str(DEFAULT_PROMPT).endswith("prompt_qwen36_synthetic_generator.yaml"))
        self.assertTrue(str(TASK_PROMPT).endswith("prompt_qwen36_angular2_nothink.yaml"))

    def test_generation_message_is_gold_conditioned(self):
        message = build_backtracking_user_message("Who directed X?", "Danny Boyle")
        self.assertIn("Reference answer: Danny Boyle", message)
        self.assertIn("<fact> <Subject> <relation>", message)

    def test_generation_message_has_example(self):
        message = build_backtracking_user_message("Who directed X?", "Danny Boyle")
        self.assertIn("Example of the required shape", message)
        self.assertIn("<fact> <Slumdog Millionaire> <producer>", message)
        self.assertIn("I reject this candidate and back out of this branch", message)
        # The example must teach the open-tag contract, not write its own object.
        self.assertNotIn("</fact>", message)


class TestParsing(unittest.TestCase):

    def test_parse_fact_calls(self):
        calls = parse_fact_calls(GOOD_TRACE)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["relation"], "description")
        self.assertEqual(calls[1]["object"], "Danny Boyle")
        self.assertFalse(calls[0]["sentinel"])

    def test_parse_open_fact_calls(self):
        text = "<fact> <France> <shares border with>\n"
        calls = parse_open_fact_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["subject"], "France")
        self.assertEqual(calls[0]["relation"], "shares border with")
        # A closed call is not pending.
        self.assertEqual(parse_open_fact_calls("<fact> <a> <b> <c> . </fact>"), [])

    def test_inject_fact_result(self):
        text = "<fact> <France> <capital>\n"
        call = parse_open_fact_calls(text)[-1]
        injected = inject_fact_result(text, call, "Paris")
        self.assertIn("<France> <capital> <Paris> . </fact>", injected)
        self.assertEqual(parse_fact_calls(injected)[0]["object"], "Paris")

    def test_inject_sentinel(self):
        text = "<fact> <France> <capital>\n"
        call = parse_open_fact_calls(text)[-1]
        injected = inject_fact_result(text, call, None)
        self.assertTrue(parse_fact_calls(injected)[0]["sentinel"])


class TestValidation(unittest.TestCase):

    def test_good_trace(self):
        ok, reasons, diagnostics = validate_backtracking_trace(
            GOOD_TRACE, RECORD, dataset="mintaka")
        self.assertTrue(ok, reasons)
        self.assertEqual(diagnostics["wrong_candidate"], "2008 film")
        self.assertTrue(diagnostics["wrong_branch_closed"])
        self.assertTrue(diagnostics["grounded"])

    def test_no_correction(self):
        trace = GOOD_TRACE.replace(
            "That's not the right path: I reject this candidate and back out of this branch.", ""
        ).replace("I will instead retrieve", "I will retrieve")
        ok, reasons, _ = validate_backtracking_trace(trace, RECORD, dataset="mintaka")
        self.assertFalse(ok)
        self.assertTrue(any("correction" in r for r in reasons), reasons)

    def test_no_recovery(self):
        trace = GOOD_TRACE.replace(
            "<fact> <Slumdog Millionaire> <director> <Danny Boyle> . </fact>\n", ""
        )
        ok, reasons, _ = validate_backtracking_trace(trace, RECORD, dataset="mintaka")
        self.assertFalse(ok)
        self.assertTrue(any("recovery" in r for r in reasons), reasons)

    def test_open_wrong_branch(self):
        trace = GOOD_TRACE.replace(
            "That's not the right path: I reject this candidate and back out of this branch.",
            "This is not the right answer.",
        )
        ok, reasons, diagnostics = validate_backtracking_trace(trace, RECORD, dataset="mintaka")
        self.assertFalse(ok)
        self.assertFalse(diagnostics["wrong_branch_closed"])
        self.assertTrue(any("not closed" in r for r in reasons), reasons)

    def test_ungrounded_fact(self):
        trace = GOOD_TRACE.replace("<Danny Boyle>", "<Someone Else>")
        ok, reasons, diagnostics = validate_backtracking_trace(trace, RECORD, dataset="mintaka")
        self.assertFalse(ok)
        self.assertFalse(diagnostics["grounded"])
        self.assertTrue(any("ungrounded" in r for r in reasons), reasons)

    def test_wrong_answer(self):
        trace = GOOD_TRACE.replace('["Danny Boyle"]', '["Christopher Nolan"]')
        ok, reasons, _ = validate_backtracking_trace(trace, RECORD, dataset="mintaka")
        self.assertFalse(ok)
        self.assertTrue(any("not correct" in r for r in reasons), reasons)

    def test_rejected_from_trace(self):
        ok, _, diagnostics = validate_backtracking_trace(GOOD_TRACE, RECORD, dataset="mintaka")
        self.assertTrue(ok)
        rejected = rejected_from_trace(GOOD_TRACE, diagnostics)
        self.assertIsNotNone(rejected)
        self.assertIn('["2008 film"]', rejected)
        self.assertNotIn("That's not the right path", rejected)


class TestInjection(unittest.TestCase):

    def test_build_injected_trace(self):
        result = build_injected_trace(RECORD)
        self.assertTrue(result["ok"], result["reason"])
        self.assertEqual(result["wrong_candidate"], "2008 film")
        self.assertIn("Danny Boyle", result["trace"])
        ok, reasons, _ = validate_backtracking_trace(result["trace"], RECORD, dataset="mintaka")
        self.assertTrue(ok, reasons)

    def test_injected_dpo_pair(self):
        result = inject_one(RECORD, dataset="mintaka")
        self.assertTrue(result["ok"], result["reason"])
        rejected = result["rejected_trace"]
        self.assertIn("2008 film", rejected)
        # The rejected completion answers the wrong candidate, the chosen one does not.
        self.assertIn('["2008 film"]', rejected)
        self.assertNotIn("Danny Boyle", rejected)

    def test_needs_two_triples(self):
        record = dict(RECORD, triples=[" <Slumdog Millionaire> <director> <Danny Boyle> ."])
        self.assertFalse(build_injected_trace(record)["ok"])

    def test_needs_distractor(self):
        record = dict(RECORD, triples=[" <Slumdog Millionaire> <director> <Danny Boyle> .",
                                       " <Other> <director> <Danny Boyle> ."])
        self.assertFalse(build_injected_trace(record)["ok"])


class TestFactOracle(unittest.TestCase):

    def test_serves_then_exhausts(self):
        oracle = SourceFactOracle(RECORD)
        first = oracle("Slumdog Millionaire", "description")
        self.assertEqual(first["object"], "2008 film")
        second = oracle("Slumdog Millionaire", "description")
        self.assertTrue(second["complete"])
        self.assertIsNone(second["object"])

    def test_unknown_relation(self):
        oracle = SourceFactOracle(RECORD)
        result = oracle("Slumdog Millionaire", "cast member")
        self.assertTrue(result["unknown"])
        self.assertTrue(result["complete"])


class TestRecordAndCLI(unittest.TestCase):

    def test_build_record(self):
        result = inject_one(RECORD, dataset="mintaka")
        sample = build_record(RECORD, RECORD["question"], result, system_prompt="system",
                             generator_meta={"type": "inject"}, dataset="mintaka")
        self.assertTrue(sample["synthetic"])
        self.assertEqual(sample["prediction"], ["Danny Boyle"])
        self.assertEqual(sample["evaluation"]["correct"], 1)
        self.assertIn("backtracking", sample["behaviour_tags"])
        self.assertTrue(sample["verification"]["ok"])
        self.assertIsNotNone(sample["dpo_pair"])
        # The trained prompt is inference-style: no behavioural instruction.
        self.assertEqual(sample["prompt"], "system\n\nQuestion: Who directed Slumdog Millionaire?")
        self.assertEqual(sample["triples"][0].strip(), "<Slumdog Millionaire> <description> <2008 film> .")

    def _write_log(self, path):
        header = {"dataset": "rpozzi/mintaka", "dataset_split": "train"}
        records = [
            dict(RECORD, question="q1"),
            dict(RECORD, question="q2", reached_max_tokens=True),
            dict(RECORD, question="q3", evaluation={"correct": 0}),
            dict(RECORD, question="q4", triples=[" <a> <b> <c> ."]),
        ]
        with path.open("w") as f:
            f.write(json.dumps(header) + "\n")
            for record in records:
                f.write(json.dumps(record) + "\n")
            f.write(json.dumps({"macro_precision": 1.0}) + "\n")

    def test_read_candidates_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.out"
            self._write_log(path)
            candidates = read_candidates(path, select="positive")
            self.assertEqual([c["question"] for c in candidates], ["q1"])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.out"
            output = Path(tmp) / "backtracking.jsonl"
            self._write_log(path)
            main(["--input", str(path), "--output", str(output), "--n", "1", "--quiet"])
            lines = [json.loads(line) for line in output.read_text().splitlines()]
            header, records, summary = lines[0], lines[1:-1], lines[-1]
            self.assertEqual(header["behaviour"], "backtracking")
            self.assertTrue(header["synthetic"])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["prediction"], ["Danny Boyle"])
            self.assertTrue(records[0]["synthetic"])
            self.assertIn("dpo_pair", records[0])
            self.assertTrue(summary["synthetic_backtracking"])

    def test_cli_generate_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.out"
            output = Path(tmp) / "backtracking.jsonl"
            self._write_log(path)
            main(["--input", str(path), "--output", str(output), "--n", "1",
                  "--mode", "generate", "--mock", "--quiet"])
            records = [json.loads(line) for line in output.read_text().splitlines()][1:-1]
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["verification"]["ok"])


if __name__ == "__main__":
    unittest.main()
