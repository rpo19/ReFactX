from base_postgres_index_config import IndexConfig

index_config = IndexConfig(
    model_name = 'microsoft/phi-4',
    # postgresql_url=None, # from dotenv
    # cache=None, # from dotenv
    postgresql_table = 'ctriev5phi4',
    switch_parameter = 7,
    rootkey = 200000,
    end_of_triple = 662,
    cache_db=2,
)

