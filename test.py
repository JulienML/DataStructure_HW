from filter import compute_filter_query_costs, compute_join_query_costs

result = compute_filter_query_costs(
    database="db1",
    collection="Stock",
    output_keys=["quantity", "location"],
    filter_keys=["IDP", "IDW"],
    sharding=True,
    sharding_key="IDP",
)

print(result)

result2 = compute_join_query_costs(
    database="db1",
    collections=["Product", "Stock"],
    output_keys={"Stock": ["IDW", "quantity"], "Product": ["name", "price"]},
    join_keys={"Stock": "IDP", "Product": "IDP"},
    filter_keys={"Stock": [], "Product": ["brand"]},
    sharding={"Stock": True, "Product": True},
    sharding_keys={"Stock": "IDP", "Product": "brand"},
)

print(result2)