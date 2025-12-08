def sharding_distribution(
        nb_documents: int,
        nb_distinct_values: int,
        servers: int
    ):
    return {
        "docs_per_server": nb_documents / servers,
        "distinct_values_per_server": nb_distinct_values / servers
    }
