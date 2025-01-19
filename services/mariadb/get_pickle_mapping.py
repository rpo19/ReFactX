import mariadb
import pickle

# Database connection details
db_config = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "example",
    "database": "mysql"
}

# Query to retrieve data
query = """
SELECT pp_value AS wikidata_qid, page_title
FROM page, page_props
WHERE page_id = pp_page AND pp_propname = 'wikibase_item';
"""

# Connect to the database and execute the query
connection = mariadb.connect(**db_config)
cursor = connection.cursor()

# Execute the query
cursor.execute(query)

# Populate the dictionary
wikidata_dict = {row[0]: row[1] for row in cursor.fetchall()}

print(f"Retrieved {len(wikidata_dict)} entries.")
print(list(wikidata_dict.items())[:5])
with open('wikidata_titles_mapping.pickle', 'wb') as fd:
    pickle.dump(wikidata_dict, fd)

cursor.close()
connection.close()

