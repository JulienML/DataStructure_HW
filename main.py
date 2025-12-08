import os
from loader import load_json_schema, load_statistics
from model import CollectionSchema
from size import estimate_document_size, estimate_collection_size, gb
from sharding import sharding_distribution

def load_collections_from_folder(folder):
    collections = {}
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            path = os.path.join(folder, filename)
            js = load_json_schema(path)
            name = filename.replace(".json", "")
            collections[name] = CollectionSchema(name, js)
    return collections


def compute_db_stats(db_name, folder, docs_per_collection, distinct_values_map):
    print("\n======================================================")
    print(f"                DATABASE {db_name}")
    print("======================================================")

    schemas = load_collections_from_folder(folder)
    stats = load_statistics()
    server_count = stats["servers"]

    total_db_size_bytes = 0

    for coll_name, schema in schemas.items():

        print(f"\n--- Collection: {coll_name} ---")

        doc_size = estimate_document_size(schema)
        print(f"Document size: {doc_size} Bytes")

        nb_docs = docs_per_collection.get(coll_name, 0)
        coll_size = estimate_collection_size(doc_size, nb_docs)
        total_db_size_bytes += coll_size
        print(f"Collection size: {gb(coll_size)} GB")

        distinct_vals = distinct_values_map.get(coll_name, 1)
        shard = sharding_distribution(nb_docs, distinct_vals, server_count)
        print(f"Sharding → docs/server: {shard['docs_per_server']}, "
              f"distinct/server: {shard['distinct_values_per_server']}")

    print("\n=== TOTAL DATABASE SIZE ===")
    print(f"{gb(total_db_size_bytes)} GB")
    print("======================================================\n")

if __name__ == "__main__":

    docs = {
        "Product": 100000,
        "Stock": 4000000000,
        "Warehouse": 200,
        "OrderLine": 4000000000,
        "Client": 10000000
    }

    distinct = {
        "Product": 5000,       
        "Stock": 100000,         
        "Warehouse": 200,
        "OrderLine": 365,        
        "Client": 10000000
    }

    # DB1
    compute_db_stats(
        db_name="DB1",
        folder="schemas/db1",
        docs_per_collection=docs,
        distinct_values_map=distinct
    )

    # DB2
    compute_db_stats(
        db_name="DB2",
        folder="schemas/db2",
        docs_per_collection=docs,
        distinct_values_map=distinct
    )

    # DB3
    compute_db_stats(
        db_name="DB3",
        folder="schemas/db3",
        docs_per_collection=docs,
        distinct_values_map=distinct
    )

    # DB4
    compute_db_stats(
        db_name="DB4",
        folder="schemas/db4",
        docs_per_collection=docs,
        distinct_values_map=distinct
    )

    # DB5
    compute_db_stats(
        db_name="DB5",
        folder="schemas/db5",
        docs_per_collection=docs,
        distinct_values_map=distinct
    )

    print("\n========== ORIGINAL PRODUCT TEST ==========\n")

    js = load_json_schema("schemas/Product.json")
    schema = CollectionSchema("Product", js)

    doc_size = estimate_document_size(schema)
    coll_size = estimate_collection_size(doc_size, 100000)

    print("=== SIZE RESULTS ===")
    print("Document size (Bytes):", doc_size)
    print("Collection size (GB):", gb(coll_size))

    shard = sharding_distribution(100000, 5000, load_statistics()["servers"])
    print("\n=== SHARDING RESULTS ===")
    print(shard)
