import sys
import json

"""
Fix judge in case of modified "yes policy".
"""

if len(sys.argv) < 2:
    print("Usage: python fix_judge.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]
judge_evaluation = []

with open(file_path, 'r') as judge_fd:
    judge_line = judge_fd.readline()
    while judge_line:
        sample = json.loads(judge_line)
        sample['llm_decision'] = 'yes' if sample['llm_full_answer'].lower().startswith('yes') else 'no'
        judge_evaluation.append(sample)
        judge_line = judge_fd.readline()

with open(file_path, 'w') as judge_fd:
    for sample in judge_evaluation:
        judge_fd.write(json.dumps(sample) + '\n')