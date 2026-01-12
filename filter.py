from loader import load_schemas_from_folder
from sharding import compute_sharding_distribution
from size import get_all_properties, get_custom_doc_size

from settings import NB_DOCS, NB_SERVERS, STATISTICS, COST_INFOS, KEY_SIZE, VALUE_SIZES

from pathlib import Path

def compute_filter_query_costs(
    database: str,
    collection: str,
    output_keys: list[str],
    filter_keys: list[str],
    sharding: bool = False,
    sharding_key: str | None = None,
):
    # Import schemas
    schemas_path = Path(f"schemas/{database}")
    schemas = load_schemas_from_folder(schemas_path)

    # Validate collection
    if collection not in schemas:
        raise ValueError(f"Collection '{collection}' not found in schemas")
    
    # Validate output keys
    schema = schemas[collection]
    properties = get_all_properties(schema).keys()
    for key in output_keys:
        if key not in properties:
            available_keys = ", ".join(properties)
            raise ValueError(
                f"Output key '{key}' is not a property of collection '{collection}'. "
                f"Available properties: {available_keys}"
            )
    
    # Validate filter keys
    for key in filter_keys:
        if key not in properties:
            available_keys = ", ".join(properties)
            raise ValueError(
                f"Filter key '{filter_keys}' is not a property of collection '{collection}'. "
                f"Available properties: {available_keys}"
            )

    # Validate sharding key
    if sharding:
        if not sharding_key:
            raise ValueError("Sharding key must be provided when sharding is enabled")
        if sharding_key not in properties:
            available_keys = ", ".join(properties)
            raise ValueError(
                f"Sharding key '{sharding_key}' is not a property of collection '{collection}'. "
                f"Available properties: {available_keys}"
            )
    
    # Compute number of servers checked
    if sharding:
        sharding_distribution = compute_sharding_distribution(collection, sharding_key)
        docs_per_server = sharding_distribution["docs_per_server"]
        if sharding_key in filter_keys:
            nb_servers_checked = 1
        else:
            nb_servers_checked = sharding_distribution["nb_servers_used"]
    else:
        nb_servers_checked = NB_SERVERS

    # Compute number of output documents
    nb_docs = NB_DOCS.get(collection, 0)
    distinct_items = 1
    for filter_key in filter_keys:
        stat = f"distinct_{filter_key}"
        if not stat.endswith("s"):
            stat += "s"
        nb_distinct_values = STATISTICS.get(stat, nb_docs)
        distinct_items *= nb_distinct_values
    
    nb_output_docs = nb_docs // min(distinct_items, nb_docs)

    # Compute size of scanned data
    scanned_doc_size = get_custom_doc_size(schema, keys=set(filter_keys + output_keys))
    if sharding:
        scanned_data_size = scanned_doc_size * docs_per_server * nb_servers_checked
    else:
        scanned_data_size = scanned_doc_size * nb_docs
    
    # Compute size of output data
    output_doc_size = get_custom_doc_size(schema, keys=set(output_keys))
    output_data_size = output_doc_size * nb_output_docs

    # Compute costs
    total_data_size = scanned_data_size + output_data_size
    time_cost = total_data_size / COST_INFOS["bandwidth"]
    carbon_footprint = total_data_size * COST_INFOS["carbon_footprint"]
    price_cost = total_data_size * COST_INFOS["price"]
    
    return {
        "nb_servers_checked": nb_servers_checked,
        "nb_output_docs": nb_output_docs,
        "scanned_doc_byte_size": scanned_doc_size,
        "scanned_data_byte_size": scanned_data_size,
        "output_doc_byte_size": output_doc_size,
        "output_data_byte_size": output_data_size,
        "time_cost_seconds": time_cost,
        "carbon_footprint_gCO2": carbon_footprint,
        "price_cost_$": price_cost
    }

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

    collection1_costs = compute_filter_query_costs(
        database=database,
        collection=collections[0],
        output_keys=output_keys[collections[0]] + [join_keys[collections[0]]],
        filter_keys=filter_keys[collections[0]],
        sharding=sharding[collections[0]],
        sharding_key=sharding_keys[collections[0]] if sharding[collections[0]] else None,
    )

    collection2_costs = compute_filter_query_costs(
        database=database,
        collection=collections[1],
        output_keys=output_keys[collections[1]],
        filter_keys=filter_keys[collections[1]] + [join_keys[collections[1]]],
        sharding=sharding[collections[1]],
        sharding_key=sharding_keys[collections[1]] if sharding[collections[1]] else None,
    )

    total_costs = {
        "nb_servers_checked": {
            collections[0]: collection1_costs["nb_servers_checked"],
            collections[1]: collection2_costs["nb_servers_checked"]
        },
        "nb_output_docs": {
            collections[0]: collection1_costs["nb_output_docs"],
            collections[1]: collection2_costs["nb_output_docs"]
        },
        "scanned_doc_byte_size": {
            collections[0]: collection1_costs["scanned_doc_byte_size"],
            collections[1]: collection2_costs["scanned_doc_byte_size"]
        },
        "scanned_data_byte_size": {
            collections[0]: collection1_costs["scanned_data_byte_size"],
            collections[1]: collection2_costs["scanned_data_byte_size"]
        },
        "output_doc_byte_size": {
            collections[0]: collection1_costs["output_doc_byte_size"],
            collections[1]: collection2_costs["output_doc_byte_size"]
        },
        "output_data_byte_size": {
            collections[0]: collection1_costs["output_data_byte_size"],
            collections[1]: collection2_costs["output_data_byte_size"]
        },
        "time_cost_seconds": collection1_costs["time_cost_seconds"] + collection1_costs["nb_output_docs"] * collection2_costs["time_cost_seconds"],
        "carbon_footprint_gCO2": collection1_costs["carbon_footprint_gCO2"] + collection1_costs["nb_output_docs"] * collection2_costs["carbon_footprint_gCO2"],
        "price_cost_$": collection1_costs["price_cost_$"] + collection1_costs["nb_output_docs"] * collection2_costs["price_cost_$"]
    }

    return total_costs