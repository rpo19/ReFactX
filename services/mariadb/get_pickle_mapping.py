import pymysql
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
SELECT wikidata_number, page_title, wikibase_shortdesc FROM mappings_plus WHERE wikidata_prefix = 'Q'; -- Only for entities
"""

# Connect to the database and execute the query
connection = pymysql.connect(**db_config)
cursor = connection.cursor()

# Execute the query
cursor.execute(query)

# Populate the dictionary
wikidata_dict = {row[0]: {'title': row[1].decode(), 'short_desc': row[2].decode()} for row in cursor.fetchall()}

print(f"Retrieved {len(wikidata_dict)} entries.")
print(list(wikidata_dict.items())[:5])
with open('wikidata_titles_mapping.pickle', 'wb') as fd:
    pickle.dump(wikidata_dict, fd)

cursor.close()
connection.close()

