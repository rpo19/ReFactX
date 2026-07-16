#!/usr/bin/env python3
"""Populate a PostgreSQL prefix-tree index with entity names from entity_names.txt.gz."""
import gzip, os, sys
from dotenv import load_dotenv
from transformers import AutoTokenizer
import refactx

ENTITY_NAMES = "/workspace/data/entity_names.txt.gz"
TABLENAME = "entity_index"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

load_dotenv()
POSTGRES_BASE_URL = os.environ.get("POSTGRES_BASE_URL")
assert POSTGRES_BASE_URL, "POSTGRES_BASE_URL not set in .env or environment"
# Ensure database is in the URL (env URL has trailing / without db name)
if POSTGRES_BASE_URL.endswith("/"):
    POSTGRES_BASE_URL += "postgres"

print(f"Loading tokenizer {MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print(f"Opening {ENTITY_NAMES} ...")
reader = gzip.open(ENTITY_NAMES, "rb")

print("Populating PostgreSQL index ... (this may take a while)")
refactx.populate_postgres_index(
    reader,
    POSTGRES_BASE_URL,
    tokenizer,
    TABLENAME,
    batch_size=5000,
    rootkey=-100,
    configkey=-200,
    switch_parameter=7,
    total_number_of_triples=None,
    prefix="",
    tokenizer_batch_size=5000,
    add_special_tokens=False,
    count_leaves=True,
    debug=False,
)

print("Done populating index.")
