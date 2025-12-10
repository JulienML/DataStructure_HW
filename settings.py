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
    "avg_categories": 2,
    "avg_orderlines": 4*10**4,
    "avg_stocks": 200,

    "distinct_IDPs": 10**5,
    "distinct_IDWs": 200,
    "distinct_IDCs": 10**7,
    "distinct_brands": 5000,
    "distinct_dates": 365
}

COST_INFOS = {
    "bandwidth": 10**6, # 1 MB/s
    "carbon_footprint": 1000 * 10**-9, # 1000 gCO2/GB
    "price": 10**-10 # $0.0000000001/B ($0.10/GB)
}