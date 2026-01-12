from filter import compute_filter_query_costs, compute_join_query_costs
from pprint import pprint
from aggregate import compute_aggregate_query_costs

"""
compute the cost of this query :
SELECT S.quantity, S.location
FROM Stock S
WHERE S.IDP = $IDP AND S.IDW = $IDW;

on a sharded database on IDP
"""
result = compute_filter_query_costs(
    database="db1",
    collection="Stock",
    output_keys=["quantity", "location"],
    filter_keys=["IDP", "IDW"],
    sharding=True,
    sharding_key="IDP",
)

pprint(result)


"""
compute the cost of this query :
SELECT P.name, P.price, S.IDW, S.quantity
FROM Product P JOIN Stock S ON P.IDP = S.IDP
WHERE P.brand = "Apple";
"""

result2 = compute_join_query_costs(
    database="db1",
    collections=["Product", "Stock"],
    output_keys={"Stock": ["IDW", "quantity"], "Product": ["name", "price"]},
    join_keys={"Stock": "IDP", "Product": "IDP"},
    filter_keys={"Stock": [], "Product": ["brand"]},
    sharding={"Stock": True, "Product": True},
    sharding_keys={"Stock": "IDP", "Product": "brand"},
)

pprint(result2)


"""
compute the cost of this query :
SELECT P.name, P.price, OL.NB
FROM Product P JOIN (
    SELECT O.IDP, SUM(O.quantity) AS NB
    FROM OrderLine O
    GROUP BY O.IDP
) OL ON P.IDP = O.IDP
ORDER BY OL.NB DESC
LIMIT 100;
"""

# result3 = compute_aggregate_query_costs(
#     database="db1",
#     collections=["OrderLine", "Product"],  # First collection is the one with GROUP BY
#     output_keys={"OrderLine": [], "Product": ["name", "price"]},  # OL.NB is computed, P.name and P.price from Product
#     join_keys={"OrderLine": "IDP", "Product": "IDP"},
#     filter_keys={"OrderLine": [], "Product": []},
#     groupby_keys={"OrderLine": ["IDP"], "Product": []},
#     aggregate_keys={"OrderLine": ["quantity"], "Product": []},  # SUM(O.quantity)
#     sharding={"OrderLine": True, "Product": True},
#     sharding_keys={"OrderLine": "IDP", "Product": "brand"},
#     limit=100,
#     detailed=False  # Set to True for detailed phase breakdown
# )

# print("\n" + "="*50)
# print("Aggregate Query Costs:")
# print("="*50)
# pprint(result3)




print("\n" + "="*50)
print("Aggregate Query Costs with Filter:")
print("="*50)
"""
compute the cost of this query :
SELECT P.name, P.price, OL.NB
    FROM Product P JOIN (
    SELECT O.IDP, SUM(O.quantity) AS NB
    FROM OrderLine O
    WHERE O.IDC = 125
    GROUP BY O.IDP
) OL ON P.IDP = OL.IDP
ORDER BY OL.NB DESC
LIMIT 1;
"""

result4 = compute_aggregate_query_costs(
    database="db1",
    collections=["OrderLine", "Product"],  # First collection is the one with GROUP BY
    output_keys={"OrderLine": [], "Product": ["name", "price"]},  # OL.NB is computed, P.name and P.price from Product
    join_keys={"OrderLine": "IDP", "Product": "IDP"},
    filter_keys={"OrderLine": ["IDC"], "Product": []},
    groupby_keys={"OrderLine": ["IDP"], "Product": []},
    aggregate_keys={"OrderLine": ["quantity"], "Product": []},  # SUM(O.quantity)
    sharding={"OrderLine": True, "Product": True},
    sharding_keys={"OrderLine": "IDP", "Product": "IDP"},
    limit=1,
    detailed=False  # Set to True for detailed phase breakdown
)

pprint(result4)
