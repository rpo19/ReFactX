from bi_index import IndexConfig

index_config = IndexConfig('microsoft/phi-4')
index_config.load_from_path('qa_triples.zip')
index_config.clean()
