# CREATE TABLE my_table (
#     id BIGINT, -- later make it PRIMARY KEY
#     key BYTEA NOT NULL,
#     child BYTEA NOT NULL,
#     child_id BIGINT NOT NULL
# );

import psycopg
import random

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

rootkey = 60000
enroot = tken(rootkey)

sentence = [enroot]

while True:
    
    cur.execute('SELECT children FROM ctrie WHERE key = %s;', (b''.join(sentence),))
    res = cur.fetchall()

    if len(res) > 0:
    
        exploded_children = set()
        for children in res:
            children = children[0]
            exploded_children.update(set(children[i:i+2] for i in range(0, len(children), 2)))
        
        next_token = random.choice(list(exploded_children))
        
        sentence.append(next_token)
        
        print(list(map(tkde,sentence)))
    else:
        print('.')
        break
