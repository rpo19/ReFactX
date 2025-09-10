from bi_index import IndexConfig

index_config = IndexConfig('Qwen/Qwen2.5-1.5B-Instruct')
index_config.load_from_path('qa_triples.zip')
index_config.clean()