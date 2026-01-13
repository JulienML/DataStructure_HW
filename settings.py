NB_DOCS = {
    "Product": 10**5,
    "Stock": 200*10**5, # Stocks = products * warehouses
    "Warehouse": 200,
    "OrderLine": 4*10**9,
    "Client": 10**7
}

NB_SERVERS = 1000

KEY_SIZE = 12
VALUE_SIZES = {
    "number": 8,
    "integer": 8,
    "string": 80,
    "date": 20,
    "longstring": 200
}

STATISTICS = {
    "avg_categories_by_product": 2,
    "avg_orderlines_by_product": 4*10**4,
    "avg_stocks_by_product": 200,
    "avg_orderlines_by_client": 100,
    "avg_products_by_client": 20,

    "distinct_IDPs": 10**5,
    "distinct_IDWs": 200,
    "distinct_IDCs": 10**7,
    "distinct_brands": 5000,
    "distinct_dates": 365
}

PRIMARY_KEYS_TABLE = {
    "IDP": "Product",
    "IDW": "Warehouse",
    "IDC": "Client",
    "IDS": "Supplier",
}

COST_INFOS = {
    "bandwidth": 100*10**6, # 100 MB/s
    "carbon_footprint": 0.011*10**-9, # 0.011 kgCO2eq/GB
    "price": 0.011*10**-9 # 0.011€/GB
}