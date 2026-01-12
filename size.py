from loader import load_schemas_from_folder

from settings import NB_DOCS, KEY_SIZE, VALUE_SIZES, STATISTICS
from pathlib import Path

def estimate_doc_size(schema: dict, table_title: str = '') -> int:
    total_size = 0
    if table_title == '':
        table_title = schema.get("title", table_title)
    properties = schema.get("properties", {})
    for field_name, field_props in properties.items():        
        ftype = field_props.get("type")
        total_size += KEY_SIZE  # Size of the key

        if ftype in VALUE_SIZES:
            if ftype == "string" and field_props.get("format"): # Handle string formats (date and longstring)
                total_size += VALUE_SIZES[field_props["format"]]
            else:
                total_size += VALUE_SIZES[ftype]

        elif ftype == "object":
            total_size += estimate_doc_size(field_props)

        elif ftype == "array":
            nb_items = STATISTICS.get(f"avg_{field_name}_by_{table_title.lower()}", 1)
            items = field_props.get("items", {})
            total_size += estimate_doc_size(items) * nb_items

    return total_size

def compute_db_size(schemas_folder_path: str | Path) -> dict:
    if not isinstance(schemas_folder_path, Path):
        schemas_folder_path = Path(schemas_folder_path)

    results = {}
    
    total_db_size = 0
    schemas = load_schemas_from_folder(schemas_folder_path)

    for coll_name, schema in schemas.items():
        results[coll_name] = {}
        doc_size = estimate_doc_size(schema)
        results[coll_name]['document_byte_size'] = doc_size
        
        nb_docs = NB_DOCS.get(coll_name, 0)
        coll_size = doc_size * nb_docs

        results[coll_name]['collection_byte_size'] = coll_size

        total_db_size += coll_size

    results["total_database_byte_size"] = total_db_size
    
    return results

def get_all_properties(schema:dict) -> list[str]:
    properties_dict = {}

    properties = schema.get("properties", {})
    for field_name, field_props in properties.items():
        ftype = field_props.get("type")
        if ftype == "object":
            nested_props = get_all_properties(field_props)
            properties_dict.update(nested_props)
        elif ftype == "array":
            items = field_props.get("items", {})
            if items.get("type") == "object":
                nested_props = get_all_properties(items)
                properties_dict.update(nested_props)
            else:
                if items.get("type") == "string" and items.get("format"):
                    ftype = items["format"]
                
                properties_dict.update({field_name: ftype})
        else:
            if ftype == "string" and field_props.get("format"):
                ftype = field_props["format"]
            properties_dict.update({field_name: ftype})
        
    return properties_dict

def get_custom_doc_size(
    schema: dict,
    keys: set[str]
) -> int:
    total_size = 0
    properties = get_all_properties(schema)

    for key in keys:
        if key not in properties:
            raise ValueError(f"Key '{key}' not found in schema properties")
        
        ftype = properties[key]
        total_size += KEY_SIZE  # Size of the key

        if ftype in VALUE_SIZES:
            total_size += VALUE_SIZES[ftype]
        else:
            raise ValueError(f"Unsupported field type '{ftype}' for key '{key}'")
    
    return total_size