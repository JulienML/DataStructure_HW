from utils.loader import load_schemas_from_folder
from utils.size import get_all_properties, get_custom_doc_size
from utils.sharding import compute_sharding_distribution
from .filter import compute_filter_query_costs

from settings import NB_DOCS, COST_INFOS, NB_SERVERS, STATISTICS, PRIMARY_KEYS_TABLE

from pathlib import Path

def compute_aggregate_query_costs(
    database: str,
    collections: list[str],
    output_keys: dict[str, list[str]],
    join_keys: dict[str, str],
    group_by_keys: dict[str, str],
    filter_keys: dict[str, list[str]],
    sharding: dict[str, bool],
    sharding_keys: dict[str, str],
    limit: int | None = None
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
        
        # Validate group by keys
        if group_by_keys[collection] and group_by_keys[collection] not in properties:
            available_keys = ", ".join(properties)
            raise ValueError(
                f"Group by key '{group_by_keys[collection]}' is not a property of collection '{collection}'. "
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

    inner_collection = collections[1]
    outer_collection = collections[0]

    # ---------------------------
    # INNER COLLECTION AGGREGATE
    # ---------------------------

    inner_output_keys = output_keys[inner_collection] + [group_by_keys[inner_collection]] + [join_keys[inner_collection]]
    inner_filter_keys = filter_keys[inner_collection]
    inner_sharding = sharding[inner_collection]
    inner_sharding_key = sharding_keys[inner_collection] if sharding[inner_collection] else None

    # Compute number of servers checked
    if inner_sharding:
        sharding_distribution = compute_sharding_distribution(inner_collection, inner_sharding_key)
        docs_per_server = sharding_distribution["docs_per_server"]
        if inner_sharding_key in inner_filter_keys:
            nb_servers_checked = 1
        else:
            nb_servers_checked = sharding_distribution["nb_servers_used"]
    else:
        nb_servers_checked = NB_SERVERS

    # Compute number of output documents
    nb_output_docs = STATISTICS.get(f"distinct_{group_by_keys[inner_collection]}s", NB_DOCS.get(inner_collection, 0))

    avg_values_by_filter_key = []
    for filter_key in inner_filter_keys:
        if filter_key in PRIMARY_KEYS_TABLE.keys():
            filter_key = PRIMARY_KEYS_TABLE[filter_key]
        stat = f"avg_{outer_collection.lower()}s_by_{filter_key.lower()}"
        avg_values_by_filter_key.append(STATISTICS.get(stat, nb_output_docs))
    
    nb_output_docs = min(avg_values_by_filter_key + [nb_output_docs])

    # Compute shuffle
    if inner_sharding_key == group_by_keys[inner_collection] or inner_sharding_key in inner_filter_keys:
        nb_shuffles = 0
        shuffle_doc_size = 0
        shuffles_data_size = 0
    else:
        nb_shuffles = nb_servers_checked * (nb_output_docs - 1)
        # Estimate size of grouped documents
        schema = schemas[inner_collection]
        shuffle_doc_size = get_custom_doc_size(schema, keys=set(inner_output_keys))
        shuffles_data_size = shuffle_doc_size * nb_output_docs * nb_shuffles
    
    # Compute size of scanned data
    scanned_doc_size = get_custom_doc_size(schema, keys=set(inner_filter_keys + inner_output_keys))
    if sharding:
        scanned_data_size = scanned_doc_size * docs_per_server * nb_servers_checked
    else:
        scanned_data_size = scanned_doc_size * NB_DOCS.get(inner_collection, 0)
    
    # Compute size of output data
    output_doc_size = get_custom_doc_size(schema, keys=set(inner_output_keys))
    output_data_size = output_doc_size * nb_output_docs

    # Compute costs
    total_data_size = scanned_data_size + output_data_size
    time_cost = total_data_size / COST_INFOS["bandwidth"]
    carbon_footprint = total_data_size * COST_INFOS["carbon_footprint"]
    price_cost = total_data_size * COST_INFOS["price"]

    inner_collection_results = {
        "nb_servers_checked": nb_servers_checked,
        "nb_output_docs": nb_output_docs,
        "scanned_doc_byte_size": scanned_doc_size,
        "scanned_data_byte_size": scanned_data_size,
        "output_doc_byte_size": output_doc_size,
        "output_data_byte_size": output_data_size,
        "nb_shuffles": nb_shuffles,
        "shuffle_doc_byte_size": shuffle_doc_size,
        "shuffles_data_byte_size": shuffles_data_size,
        "time_cost_seconds": time_cost,
        "carbon_footprint_kgCO2eq": carbon_footprint,
        "price_cost_€": price_cost
    }

    # ---------------------------
    # OUTER COLLECTION JOIN
    # ---------------------------
    
    if limit:
        nb_loops = min(limit, inner_collection_results["nb_output_docs"])
    else:
        nb_loops = inner_collection_results["nb_output_docs"]

    outer_collection_results = compute_filter_query_costs(
        database=database,
        collection=outer_collection,
        output_keys=output_keys[outer_collection],
        filter_keys=filter_keys[outer_collection] + [join_keys[outer_collection]],
        sharding=sharding[outer_collection],
        sharding_key=sharding_keys[outer_collection] if sharding[outer_collection] else None,
    )

    # ---------------------------
    # FINAL RESULTS
    # ---------------------------

    final_results = {}

    for key in inner_collection_results:
        final_results[key] = {
            inner_collection: inner_collection_results[key],
            outer_collection: outer_collection_results.get(key, 0)
        }
    
    for cost_key in ["time_cost_seconds", "carbon_footprint_kgCO2eq", "price_cost_€"]:
        final_results[cost_key] = inner_collection_results[cost_key] + nb_loops * outer_collection_results[cost_key]
    
    final_results["nb_loops"] = nb_loops

    return final_results