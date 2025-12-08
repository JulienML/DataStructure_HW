KEY_SIZES = {
    "number": 8,
    "integer": 8,
    "string": 80,
    "date": 20,
    "longstring": 200,
    "object": 12,
    "array": 12
}

def estimate_document_size(schema):
    size = 0
    for field in schema.fields:
        size += KEY_SIZES.get(field.ftype, 12)  
    return size

def estimate_collection_size(doc_size, nb_documents):
    return doc_size * nb_documents

def gb(x):
    return x / (1024**3)
