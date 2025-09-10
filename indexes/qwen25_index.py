from .base_postgres_index_config import IndexConfig

index_config = IndexConfig(
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct',
    # postgresql_url=None, # from dotenv
    # cache=None, # from dotenv
    postgresql_table = 'ctriev6qwen',
    switch_parameter = 7,
    rootkey = 500000,
    end_of_triple = 659,
    cache_db=0,
)
