from loader import load_json_schema, load_statistics
from model import CollectionSchema
from size import estimate_document_size, estimate_collection_size, gb
from sharding import sharding_distribution

def run_test(schema_path, collection_name, nb_documents, distinct_key_values):
    stats = load_statistics()

    js = load_json_schema(schema_path)
    schema = CollectionSchema(collection_name, js)

    doc_size = estimate_document_size(schema)
    coll_size = estimate_collection_size(doc_size, nb_documents)

    print("=== SIZE RESULTS ===")
    print("Document size (Bytes):", doc_size)
    print("Collection size (GB):", gb(coll_size))

    shard = sharding_distribution(nb_documents, distinct_key_values, stats["servers"])
    print("\n=== SHARDING RESULTS ===")
    print(shard)


if __name__ == "__main__":
    run_test(
        schema_path="schemas/Product.json",
        collection_name="Product",
        nb_documents=100000,
        distinct_key_values=5000  
    )
