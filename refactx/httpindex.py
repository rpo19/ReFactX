from flask import Flask, request
import os
import importlib
from .ctrie import serialize, deserialize
from psycopg import sql

index_config_path = os.environ.get('INDEX_CONFIG_PATH')

if index_config_path.endswith('.py'):
    index_config_path = index_config_path[:-3]
index_module = importlib.import_module(index_config_path)
index_config = getattr(index_module, 'index_config')

postgresql_connection = index_config.postgresql_connection

app = Flask(__name__)

@app.route("/select", methods=["POST"])
def get_next_tokens():
    args = deserialize(request.data)
    table_name = args['table_name']
    sequence = args['sequence']
    select_query = index_config.index.base_select_query.format(sql.Identifier(table_name))

    with postgresql_connection.cursor() as cursor:
        cursor.execute(select_query, (sequence,))
        query_result = cursor.fetchall()

    return serialize(query_result)
