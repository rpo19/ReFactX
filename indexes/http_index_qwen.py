from base_http_index import IndexConfig

index_config = IndexConfig(
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct',
    # postgresql_url=None, # from dotenv
    # redis_url=None, # from dotenv
    postgresql_table = 'ctriev5qwen',
    switch_parameter = 7,
    rootkey = 200000,
    end_of_triple = 659,
    cache_db=0,
)
