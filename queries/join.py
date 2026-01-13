from utils.loader import load_schemas_from_folder, get_all_properties
from .filter import compute_filter_query_costs

from pathlib import Path

def compute_join_query_costs(
    database: str,
    collections: list[str],
    output_keys: dict[str, list[str]],
    join_keys: dict[str, str],
    filter_keys: dict[str, list[str]],
    sharding: dict[str, bool],
    sharding_keys: dict[str, str]
):
    # Import schemas
    schemas_path = Path(f"schemas/{database}")
    schemas = load_schemas_from_folder(schemas_path)

    # Validate collections
    for collection in collections:
        if collection not in schemas:
            raise ValueError(f"Collection '{collection}' not found in schemas")
    
    for collection in collections:
        schema = schemas[collection]
        properties = get_all_properties(schema).keys()

        # Validate output keys
        for key in output_keys[collection]:
            if key not in properties:
                available_keys = ", ".join(properties)
                raise ValueError(
                    f"Output key '{key}' is not a property of collection '{collection}'. "
                    f"Available properties: {available_keys}"
                )
        
        # Validate join keys
        if join_keys[collection] not in properties:
            available_keys = ", ".join(properties)
            raise ValueError(
                f"Join key '{join_keys[collection]}' not found in collection '{collection}'"
                f"Available properties: {available_keys}"
            )
        
        # Validate filter keys
        for key in filter_keys[collection]:
            if key not in properties:
                available_keys = ", ".join(properties)
                raise ValueError(
                    f"Filter key '{key}' is not a property of collection '{collection}'. "
                    f"Available properties: {available_keys}"
                )

        # Validate sharding key
        if sharding[collection]:
            if not sharding_keys[collection]:
                raise ValueError("Sharding key must be provided when sharding is enabled")
            if sharding_keys[collection] not in properties:
                available_keys = ", ".join(properties)
                raise ValueError(
                    f"Sharding key '{sharding_keys[collection]}' is not a property of collection '{collection}'. "
                    f"Available properties: {available_keys}"
                )

    collection1_results = compute_filter_query_costs(
        database=database,
        collection=collections[0],
        output_keys=output_keys[collections[0]] + [join_keys[collections[0]]],
        filter_keys=filter_keys[collections[0]],
        sharding=sharding[collections[0]],
        sharding_key=sharding_keys[collections[0]] if sharding[collections[0]] else None,
    )

    print(collection1_results["nb_output_docs"])

    collection2_results = compute_filter_query_costs(
        database=database,
        collection=collections[1],
        output_keys=output_keys[collections[1]],
        filter_keys=filter_keys[collections[1]] + [join_keys[collections[1]]],
        sharding=sharding[collections[1]],
        sharding_key=sharding_keys[collections[1]] if sharding[collections[1]] else None,
    )

    total_costs = {
        "nb_servers_checked": {
            collections[0]: collection1_results["nb_servers_checked"],
            collections[1]: collection2_results["nb_servers_checked"]
        },
        "nb_output_docs": {
            collections[0]: collection1_results["nb_output_docs"],
            collections[1]: collection2_results["nb_output_docs"]
        },
        "scanned_doc_byte_size": {
            collections[0]: collection1_results["scanned_doc_byte_size"],
            collections[1]: collection2_results["scanned_doc_byte_size"]
        },
        "scanned_data_byte_size": {
            collections[0]: collection1_results["scanned_data_byte_size"],
            collections[1]: collection2_results["scanned_data_byte_size"]
        },
        "output_doc_byte_size": {
            collections[0]: collection1_results["output_doc_byte_size"],
            collections[1]: collection2_results["output_doc_byte_size"]
        },
        "output_data_byte_size": {
            collections[0]: collection1_results["output_data_byte_size"],
            collections[1]: collection2_results["output_data_byte_size"]
        },
        "time_cost_seconds": collection1_results["time_cost_seconds"] + collection1_results["nb_output_docs"] * collection2_results["time_cost_seconds"],
        "carbon_footprint_kgCO2eq": collection1_results["carbon_footprint_kgCO2eq"] + collection1_results["nb_output_docs"] * collection2_results["carbon_footprint_kgCO2eq"],
        "price_cost_€": collection1_results["price_cost_€"] + collection1_results["nb_output_docs"] * collection2_results["price_cost_€"]
    }

    return total_costs