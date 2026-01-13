import json
from pathlib import Path

def load_schemas_from_folder(folder: str | Path = "./schemas") -> dict[str, dict]:
    if not isinstance(folder, Path):
        folder = Path(folder)
    
    schemas = {}

    for filename in folder.glob("*.json"):
        path = folder / filename.name
        with open(path, "r") as f:
            json_schema = json.load(f)
            f.close()
        name = filename.stem
        
        schemas[name] = json_schema
    
    return schemas

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