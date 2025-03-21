from base_postgres_index_config import IndexConfig

index_config = IndexConfig(
    model_name = 'google/gemma-3-4b-it',
    # postgresql_url=None, # from dotenv
    # cache=None, # from dotenv
    postgresql_table = 'ctriev5gemma3',
    switch_parameter = 7,
    rootkey = 300000,
    end_of_triple = 783,
    cache_db=3,
)
