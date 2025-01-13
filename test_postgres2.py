# CREATE TABLE my_table (
#     id BIGINT, -- later make it PRIMARY KEY
#     key BYTEA NOT NULL,
#     child BYTEA NOT NULL,
#     child_id BIGINT NOT NULL
# );

import psycopg

postgres_connection = 'postgres://postgres:secret@10.0.0.118:5432/postgres'

def tken(token):
    # token encode
    encoded = token.to_bytes(2, byteorder='big', signed=False)
    return encoded

def tkde(bbytes):
    # token decode
    decoded = int.from_bytes(bbytes, byteorder='big', signed=False)
    return decoded

conn = psycopg.connect(postgres_connection, autocommit=False)
cur = conn.cursor()
