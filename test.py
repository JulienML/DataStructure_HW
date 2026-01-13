from queries.filter import compute_filter_query_costs
from queries.join import compute_join_query_costs
from queries.aggregate import compute_aggregate_query_costs

from pprint import pprint

"""
Compute the cost of this query :

SELECT S.quantity, S.location
FROM Stock S
WHERE S.IDP = $IDP AND S.IDW = $IDW;

With Stock sharded on IDP
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
Compute the cost of this query :

SELECT P.name, P.price, S.IDW, S.quantity
FROM Product P JOIN Stock S ON P.IDP = S.IDP
WHERE P.brand = "Apple";

With Product sharded on brand and Stock sharded on IDP
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
Compute the cost of this query :

SELECT P.name, P.price, OL.NB
FROM Product P JOIN (
    SELECT O.IDP, SUM(O.quantity) AS NB
    FROM OrderLine O
    GROUP BY O.IDP
) OL ON P.IDP = O.IDP
ORDER BY OL.NB DESC
LIMIT 100;

With Product sharded on brand and OrderLine sharded on IDC
"""

result3 = compute_aggregate_query_costs(
    database="db1",
    collections=["Product", "OrderLine"],
    output_keys={"Product": ["name", "price"], "OrderLine": ["quantity"]},
    join_keys={"Product": "IDP", "OrderLine": "IDP"},
    group_by_keys={"Product": None, "OrderLine": "IDP"},
    filter_keys={"Product": [], "OrderLine": []},
    sharding={"Product": True, "OrderLine": True},
    sharding_keys={"Product": "IDP", "OrderLine": "IDC"},
    limit=100
)

pprint(result3)

"""
Compute the cost of this query :

SELECT P.name, P.price, OL.NB
FROM Product P JOIN (
    SELECT O.IDP, SUM(O.quantity) AS NB
    FROM OrderLine O
    WHERE O.idClient = 125
    GROUP BY C.IDP
) OL ON P.IDP = OL.IDP
ORDER BY OL.NB DESC
LIMIT 1;

With Product sharded on IDP and OrderLine sharded on IDP
"""

result4 = compute_aggregate_query_costs(
    database="db1",
    collections=["Product", "OrderLine"],
    output_keys={"Product": ["name", "price"], "OrderLine": ["quantity"]},
    join_keys={"Product": "IDP", "OrderLine": "IDP"},
    group_by_keys={"Product": None, "OrderLine": "IDP"},
    filter_keys={"Product": [], "OrderLine": ["IDC"]},
    sharding={"Product": True, "OrderLine": True},
    sharding_keys={"Product": "IDP", "OrderLine": "IDP"},
    limit=1
)

pprint(result4)