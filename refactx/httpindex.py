from flask import Flask, request
import os
from refactx import load_index
from refactx.index import serialize, deserialize
from psycopg import sql

index_url = os.environ.get('INDEX')
index = load_index(index_url)
postgresql_connection = index.postgresql_connection

app = Flask(__name__)

@app.route("/<table_name>", methods=["POST"])
def get_next_tokens(table_name):
    args = deserialize(request.data)
    sequence = args['sequence']
    select_query = index.base_select_query.format(sql.Identifier(table_name))

    with postgresql_connection.cursor() as cursor:
        cursor.execute(select_query, (sequence,))
        query_result = cursor.fetchall()

    return serialize(query_result)
