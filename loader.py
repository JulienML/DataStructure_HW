import json

def load_json_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)

def load_statistics():
    """
    Returns the statistics given in the PDF.
    You can adapt values later if needed.
    """
    return {
        "nb_clients": 10**7,
        "nb_products": 10**5,
        "nb_orderlines": 4 * 10**9,
        "nb_warehouses": 200,
        "servers": 1000
    }
