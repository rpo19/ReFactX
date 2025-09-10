from .simple_index import IndexConfig

index_config = IndexConfig('meta-llama/Llama-3.2-1B-Instruct')
index_config.load_from_path()
index_config.clean()