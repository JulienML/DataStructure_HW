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