
from settings import *
from sharding import compute_sharding_distribution
from size import get_all_properties, get_custom_doc_size

from loader import load_schemas_from_folder
from pathlib import Path


def compute_aggregate_query_costs(
    database: str,
    collections: list[str],
    output_keys: dict[str, list[str]],
    join_keys: dict[str, str],
    filter_keys: dict[str, list[str]],
    groupby_keys: dict[str, list[str]],
    aggregate_keys: dict[str, list[str]],
    sharding: dict[str, bool],
    sharding_keys: dict[str, str],
    limit: int | None = None,
    detailed: bool = False
):
    """
    Compute the costs of an aggregate query involving joins and group-bys.
    
    Example query:
        SELECT P.name, P.price, OL.NB
        FROM Product P JOIN (
            SELECT O.IDP, SUM(O.quantity) AS NB
            FROM OrderLine O
            GROUP BY O.IDP
        ) OL ON P.IDP = O.IDP
        ORDER BY OL.NB DESC
        LIMIT 100;
    
    Args:
        database (str): The name of the database containing the collections. corresponds to a folder in 'schemas/'.
        collections (list[str]): A list of collection names involved in the query, in the correct order.
            The first collection is the one with GROUP BY (inner subquery).
        output_keys (dict[str, list[str]]): A dictionary mapping each collection name to a list of output keys to be retrieved.
            (corresponds to fields to be projected, just after the SELECT clause)
        join_keys (dict[str, str]): A dictionary mapping each collection name to the join key used for joining with other collections.
        filter_keys (dict[str, list[str]]): A dictionary mapping each collection name to a list of filter keys used for filtering documents.
        groupby_keys (dict[str, list[str]]): A dictionary mapping each collection name to a list of keys used for grouping documents.
        aggregate_keys (dict[str, list[str]]): A dictionary mapping each collection name to a list of keys used in aggregate functions (SUM, COUNT, etc.).
        sharding (dict[str, bool]): A dictionary mapping each collection name to a boolean indicating whether the collection is sharded.
        sharding_keys (dict[str, str]): A dictionary mapping each collection name to the sharding key used if the collection is sharded.
        limit (int | None): An optional limit on the number of output documents.
        detailed (bool): If True, returns detailed breakdown by phase. If False (default), returns the same structure as compute_join_query_costs.
    
    Returns:
        dict: A dictionary containing the costs breakdown.
    """
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
        for key in output_keys.get(collection, []):
            if key not in properties:
                available_keys = ", ".join(properties)
                raise ValueError(
                    f"Output key '{key}' is not a property of collection '{collection}'. "
                    f"Available properties: {available_keys}"
                )
        
        # Validate join keys
        if collection in join_keys and join_keys[collection] not in properties:
            available_keys = ", ".join(properties)
            raise ValueError(
                f"Join key '{join_keys[collection]}' not found in collection '{collection}'. "
                f"Available properties: {available_keys}"
            )
        
        # Validate filter keys
        for key in filter_keys.get(collection, []):
            if key not in properties:
                available_keys = ", ".join(properties)
                raise ValueError(
                    f"Filter key '{key}' is not a property of collection '{collection}'. "
                    f"Available properties: {available_keys}"
                )
        
        # Validate groupby keys
        for key in groupby_keys.get(collection, []):
            if key not in properties:
                available_keys = ", ".join(properties)
                raise ValueError(
                    f"Group by key '{key}' is not a property of collection '{collection}'. "
                    f"Available properties: {available_keys}"
                )
        
        # Validate aggregate keys
        for key in aggregate_keys.get(collection, []):
            if key not in properties:
                available_keys = ", ".join(properties)
                raise ValueError(
                    f"Aggregate key '{key}' is not a property of collection '{collection}'. "
                    f"Available properties: {available_keys}"
                )

        # Validate sharding key
        if sharding.get(collection, False):
            if not sharding_keys.get(collection):
                raise ValueError(f"Sharding key must be provided when sharding is enabled for collection '{collection}'")
            if sharding_keys[collection] not in properties:
                available_keys = ", ".join(properties)
                raise ValueError(
                    f"Sharding key '{sharding_keys[collection]}' is not a property of collection '{collection}'. "
                    f"Available properties: {available_keys}"
                )


    # The first collection is the one with the GROUP BY (inner subquery)
    aggregate_collection = collections[0]
    join_collection = collections[1] if len(collections) > 1 else None
    
    aggregate_schema = schemas[aggregate_collection]
    
    # Step 1: Compute the cost of scanning and aggregating the first collection (the subquery)
    # Keys needed for the aggregate query: groupby keys + aggregate keys + filter keys
    aggregate_scan_keys = set(
        groupby_keys.get(aggregate_collection, []) + 
        aggregate_keys.get(aggregate_collection, []) + 
        filter_keys.get(aggregate_collection, [])
    )
    
    # Get number of documents and compute sharding
    nb_docs_aggregate = NB_DOCS.get(aggregate_collection, 0)
    
    if sharding.get(aggregate_collection, False):
        sharding_distribution = compute_sharding_distribution(aggregate_collection, sharding_keys[aggregate_collection])
        docs_per_server = sharding_distribution["docs_per_server"]
        
        # Check if sharding key is in filter keys (can target specific server)
        if sharding_keys[aggregate_collection] in filter_keys.get(aggregate_collection, []):
            nb_servers_checked_aggregate = 1
        else:
            nb_servers_checked_aggregate = sharding_distribution["nb_servers_used"]
        
        scanned_docs_aggregate = docs_per_server * nb_servers_checked_aggregate
    else:
        nb_servers_checked_aggregate = NB_SERVERS
        scanned_docs_aggregate = nb_docs_aggregate
    
    # Apply filter to reduce documents
    filter_selectivity = 1.0
    for filter_key in filter_keys.get(aggregate_collection, []):
        stat = f"distinct_{filter_key}"
        if not stat.endswith("s"):
            stat += "s"
        nb_distinct_values = STATISTICS.get(stat, nb_docs_aggregate)
        filter_selectivity *= (1 / nb_distinct_values)
    
    filtered_docs_aggregate = max(1, int(scanned_docs_aggregate * filter_selectivity))
    
    # Compute number of groups (output of GROUP BY)
    # Number of groups is based on distinct values of groupby keys
    nb_groups = 1
    for groupby_key in groupby_keys.get(aggregate_collection, []):
        stat = f"distinct_{groupby_key}"
        if not stat.endswith("s"):
            stat += "s"
        nb_distinct_values = STATISTICS.get(stat, filtered_docs_aggregate)
        nb_groups *= nb_distinct_values
    
    
    
    # ------------------------- JE CAPTE PAS ALED -------------------------
    # ------------------------- JE CAPTE PAS ALED -------------------------
    # ------------------------- JE CAPTE PAS ALED -------------------------
    
    # The number of output documents from the aggregate is min(filtered_docs, nb_groups)
    nb_aggregate_output_docs = min(filtered_docs_aggregate, nb_groups)
    if len(filter_keys.get(aggregate_collection, [])) > 0:
        # nb_aggregate_output_docs is equal to the avg number of documents per group after filtering
        print("filtered_docs_aggregate:", filtered_docs_aggregate)
        print("nb_groups:", nb_groups)
        avg_docs_per_group_str = f'avg_{collections[1]}_by_{aggregate_collection}' if join_collection else f'avg_documents_by_{aggregate_collection}'
        
    # ------------------------- JE CAPTE PAS ALED -------------------------
    # ------------------------- JE CAPTE PAS ALED -------------------------
    # ------------------------- JE CAPTE PAS ALED -------------------------
    
    # Compute shuffle cost (inter-server communications for aggregation)
    # If sharded key = GROUP BY key: all data for a group is on the same server → no shuffle needed
    # If NOT sharded on the GROUP BY key: need to gather partial aggregates from other servers
    groupby_keys_list = groupby_keys.get(aggregate_collection, [])
    if sharding.get(aggregate_collection, False):
        # Check if sharding key matches any of the groupby keys
        sharding_matches_groupby = sharding_keys[aggregate_collection] in groupby_keys_list
        if sharding_matches_groupby:
            # All data for each group is on the same server, no shuffle needed
            shuffle_aggregate = 0
        else:
            # Each group's data is spread across all servers
            # For each group, we need to collect partial aggregates from (nb_servers - 1) other servers
            shuffle_aggregate = nb_groups * (nb_servers_checked_aggregate - 1)
    else:
        # Not sharded, all data is on each server (replicated) or single server
        # No shuffle needed if not distributed
        shuffle_aggregate = 0
    
    # Size of scanned document for aggregate
    scanned_doc_size_aggregate = get_custom_doc_size(aggregate_schema, keys=aggregate_scan_keys)
    scanned_data_size_aggregate = scanned_doc_size_aggregate * scanned_docs_aggregate
    
    # Output of aggregate: groupby keys + one aggregated value per aggregate key
    # For simplicity, assume each aggregate result is a number (8 bytes)
    aggregate_output_keys = set(groupby_keys.get(aggregate_collection, []))
    output_doc_size_aggregate = get_custom_doc_size(aggregate_schema, keys=aggregate_output_keys)
    # Add size for each aggregate result (number type)
    output_doc_size_aggregate += len(aggregate_keys.get(aggregate_collection, [])) * (KEY_SIZE + VALUE_SIZES["number"])
    output_data_size_aggregate = output_doc_size_aggregate * nb_aggregate_output_docs
    
    # Step 2: If there's a join, compute the cost of the join
    if join_collection:
        join_schema = schemas[join_collection]
        nb_docs_join = NB_DOCS.get(join_collection, 0)
        
        if sharding.get(join_collection, False):
            sharding_distribution_join = compute_sharding_distribution(join_collection, sharding_keys[join_collection])
            docs_per_server_join = sharding_distribution_join["docs_per_server"]
            
            # Check if join key matches sharding key (can do targeted lookups)
            if sharding_keys[join_collection] == join_keys.get(join_collection):
                nb_servers_checked_join = 1
            elif sharding_keys[join_collection] in filter_keys.get(join_collection, []):
                nb_servers_checked_join = 1
            else:
                nb_servers_checked_join = sharding_distribution_join["nb_servers_used"]
            
            scanned_docs_join = docs_per_server_join * nb_servers_checked_join
        else:
            nb_servers_checked_join = NB_SERVERS
            scanned_docs_join = nb_docs_join
        
        # Keys needed to scan for the join: output keys + join key + filter keys
        join_scan_keys = set(
            output_keys.get(join_collection, []) + 
            [join_keys.get(join_collection, "")] + 
            filter_keys.get(join_collection, [])
        )
        join_scan_keys.discard("")  # Remove empty string if join key not specified
        
        scanned_doc_size_join = get_custom_doc_size(join_schema, keys=join_scan_keys)
        
        # For each aggregate output doc, we need to find matching join docs
        # Cost depends on whether join key is indexed/sharded
        if sharding.get(join_collection, False) and sharding_keys[join_collection] == join_keys.get(join_collection):
            # Optimal case: sharding key matches join key, only scan 1 server per lookup
            scanned_data_size_join = scanned_doc_size_join * docs_per_server_join
        else:
            # Need to scan all servers for each lookup (or use index if available)
            scanned_data_size_join = scanned_doc_size_join * scanned_docs_join
        
        # Apply filter to join collection
        filter_selectivity_join = 1.0
        print(f"Filter keys for join collection ({join_collection}):", filter_keys.get(join_collection, []))
        for filter_key in filter_keys.get(join_collection, []):
            stat = f"distinct_{filter_key}"
            if not stat.endswith("s"):
                stat += "s"
            nb_distinct_values = STATISTICS.get(stat, nb_docs_join)
            filter_selectivity_join *= (1 / nb_distinct_values)

        # Number of join output documents (before limit)
        # Each aggregate output doc joins with matching docs from join collection
        join_key_stat = f"distinct_{join_keys.get(join_collection, '')}"
        if not join_key_stat.endswith("s"):
            join_key_stat += "s"
        distinct_join_values = STATISTICS.get(join_key_stat, nb_docs_join)
        docs_per_join_value = max(1, nb_docs_join // distinct_join_values)

        

        # calculate nb_join_output_docs, if the id of the collection is the join key, the output is 1
        if join_keys.get(join_collection, "") == "ID" + join_collection[0].upper():
            nb_join_output_docs = 1
        else:
            nb_join_output_docs = min(nb_aggregate_output_docs * docs_per_join_value, nb_docs_join)
            nb_join_output_docs = int(nb_join_output_docs * filter_selectivity_join)
            nb_join_output_docs = max(1, nb_join_output_docs)
        
        # Output size for join collection
        output_doc_size_join = get_custom_doc_size(join_schema, keys=set(output_keys.get(join_collection, [])))
        
        # Compute shuffle cost for join
        # If sharding key matches join key: can directly target the right server → no shuffle
        # If not: for each aggregate output doc, need to query (nb_servers - 1) other servers
        if sharding.get(join_collection, False) and sharding_keys[join_collection] in get_all_properties(join_schema).keys():
            shuffle_join = 0
        elif sharding.get(join_collection, False):
            shuffle_join = nb_aggregate_output_docs * (nb_servers_checked_join - 1)
        else:
            shuffle_join = 0
    else:
        nb_servers_checked_join = 0
        scanned_doc_size_join = 0
        scanned_data_size_join = 0
        output_doc_size_join = 0
        nb_join_output_docs = nb_aggregate_output_docs
        shuffle_join = 0
    
    # Total shuffle cost
    total_shuffle = shuffle_aggregate + shuffle_join
    
    # Step 3: Apply LIMIT
    if limit is not None:
        final_output_docs = min(limit, nb_join_output_docs)
    else:
        final_output_docs = nb_join_output_docs
    
    # Final output size: combine output from aggregate and join
    if join_collection:
        # Final output includes fields from both collections
        final_output_doc_size = output_doc_size_aggregate + output_doc_size_join
    else:
        final_output_doc_size = output_doc_size_aggregate
    
    final_output_data_size = final_output_doc_size * final_output_docs
    
    # Compute total data transferred
    # For ORDER BY, we need to process all documents before applying LIMIT
    # So the scanned data includes all data before limit
    total_scanned_data = scanned_data_size_aggregate
    if join_collection:
        total_scanned_data += scanned_data_size_join
    
    total_output_data = output_data_size_aggregate + final_output_data_size
    total_data_size = total_scanned_data + total_output_data
    
    # Compute costs
    time_cost = total_data_size / COST_INFOS["bandwidth"]
    carbon_footprint = total_data_size * COST_INFOS["carbon_footprint"]
    price_cost = total_data_size * COST_INFOS["price"]
    
    # Build result based on detailed flag
    if detailed:
        result = {
            "aggregate_phase": {
                "collection": aggregate_collection,
                "nb_servers_checked": nb_servers_checked_aggregate,
                "nb_docs_scanned": int(scanned_docs_aggregate),
                "nb_groups": nb_groups,
                "nb_output_docs": nb_aggregate_output_docs,
                "shuffle": shuffle_aggregate,
                "scanned_doc_byte_size": scanned_doc_size_aggregate,
                "scanned_data_byte_size": int(scanned_data_size_aggregate),
                "output_doc_byte_size": output_doc_size_aggregate,
                "output_data_byte_size": int(output_data_size_aggregate),
            },
            "final_output": {
                "nb_docs_before_limit": nb_join_output_docs,
                "nb_docs_after_limit": final_output_docs,
                "output_doc_byte_size": final_output_doc_size,
                "output_data_byte_size": final_output_data_size,
            },
            "total_shuffle": total_shuffle,
            "total_scanned_data_byte_size": int(total_scanned_data),
            "total_output_data_byte_size": int(total_output_data),
            "time_cost_seconds": time_cost,
            "carbon_footprint_gCO2": carbon_footprint,
            "price_cost_$": price_cost
        }
        
        if join_collection:
            result["join_phase"] = {
                "collection": join_collection,
                "nb_servers_checked": nb_servers_checked_join,
                "nb_docs_scanned": int(scanned_docs_join),
                "nb_output_docs": nb_join_output_docs,
                "shuffle": shuffle_join,
                "scanned_doc_byte_size": scanned_doc_size_join,
                "scanned_data_byte_size": int(scanned_data_size_join),
                "output_doc_byte_size": output_doc_size_join,
            }
    else:
        # Same structure as compute_join_query_costs
        result = {
            "nb_servers_checked": {
                aggregate_collection: nb_servers_checked_aggregate,
            },
            "nb_output_docs": {
                aggregate_collection: nb_aggregate_output_docs,
            },
            "shuffle": {
                aggregate_collection: shuffle_aggregate,
            },
            "scanned_doc_byte_size": {
                aggregate_collection: scanned_doc_size_aggregate,
            },
            "scanned_data_byte_size": {
                aggregate_collection: int(scanned_data_size_aggregate),
            },
            "output_doc_byte_size": {
                aggregate_collection: output_doc_size_aggregate,
            },
            "output_data_byte_size": {
                aggregate_collection: int(output_data_size_aggregate),
            },
            "time_cost_seconds": time_cost,
            "carbon_footprint_gCO2": carbon_footprint,
            "price_cost_$": price_cost
        }
        
        if join_collection:
            result["nb_servers_checked"][join_collection] = nb_servers_checked_join
            result["nb_output_docs"][join_collection] = nb_join_output_docs
            result["shuffle"][join_collection] = shuffle_join
            result["scanned_doc_byte_size"][join_collection] = scanned_doc_size_join
            result["scanned_data_byte_size"][join_collection] = int(scanned_data_size_join)
            result["output_doc_byte_size"][join_collection] = output_doc_size_join
            result["output_data_byte_size"][join_collection] = int(final_output_data_size)
        
        if limit is not None:
            result["limit"] = limit
            result["nb_docs_after_limit"] = final_output_docs
        
        result["total_shuffle"] = total_shuffle
    
    return result