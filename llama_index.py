from base_postgres_index_config import IndexConfig

index_config = IndexConfig(
    model_name = 'meta-llama/Llama-3.2-1B-Instruct',
    # postgresql_url=None, # from dotenv
    # redis_url=None, # from dotenv
    postgresql_table = 'ctriev5llama',
    switch_parameter = 7,
    rootkey = 200000,
    end_of_triple = 662
)
