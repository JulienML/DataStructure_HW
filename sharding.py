from settings import STATISTICS, NB_DOCS, NB_SERVERS
from loader import load_schemas_from_folder
from pathlib import Path

def compute_sharding_distribution(collection_name: str, sharding_key: str):
    # Load the default schemas to check if sharding_key is valid
    schemas_path = Path("schemas/default")
    schemas = load_schemas_from_folder(schemas_path)
    
    # Check if the collection exists in the schemas
    if collection_name not in schemas:
        raise ValueError(f"Collection '{collection_name}' not found in schemas")
    
    # Check if the sharding_key is a property of the collection
    schema = schemas[collection_name]
    properties = schema.get("properties", {})
    
    if sharding_key not in properties:
        available_keys = ", ".join(properties.keys())
        raise ValueError(
            f"Sharding key '{sharding_key}' is not a property of collection '{collection_name}'. "
            f"Available properties: {available_keys}"
        )

    nb_values = NB_DOCS.get(collection_name, 0)
    
    # Get the number of distinct values for the sharding key
    # If not found, default to total number of documents in the collection
    stat = f"distinct_{sharding_key}"
    if not stat.endswith("s"):
        stat += "s"
    nb_distinct_values = STATISTICS.get(stat, nb_values)

    return {
        "docs_per_server": nb_values / min(NB_SERVERS, nb_distinct_values),
        "distinct_values_per_server": nb_distinct_values / min(NB_SERVERS, nb_distinct_values),
        "nb_servers_used": min(NB_SERVERS, nb_distinct_values)
    }