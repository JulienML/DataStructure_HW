import streamlit as st
from pathlib import Path

from queries.aggregate import compute_aggregate_query_costs

from utils.loader import load_schemas_from_folder
from utils.size import get_all_properties

st.set_page_config(page_title="Homework 3 - Aggregate Query Cost Computation", layout="wide")

st.title("Homework 3 - Aggregate Query Cost Computation")

tab1 = st.tabs(["Collection Aggregate"])[0]

# Helper function to get available databases
def get_available_databases():
    schemas_path = Path("schemas")
    return [d.name for d in schemas_path.iterdir() if d.is_dir() and d.name != "default"]

# Helper function to get collections from a database
def get_collections(database):
    schemas_path = Path(f"schemas/{database}")
    schemas = load_schemas_from_folder(schemas_path)
    return list(schemas.keys())

# Helper function to get properties from a collection
def get_properties(database, collection):
    schemas_path = Path(f"schemas/{database}")
    schemas = load_schemas_from_folder(schemas_path)
    if collection in schemas:
        return list(get_all_properties(schemas[collection]).keys())
    return []

# Single Collection Aggregate Query Tab
with tab1:
    st.header("Join Query Costs")
    st.write("Compute the costs of executing a join query between two collections")
    
    # Database selection
    databases_join = get_available_databases()
    default_db_join_index = databases_join.index("db1") if "db1" in databases_join else 0
    selected_db_join = st.selectbox("Select Database:", databases_join, index=default_db_join_index, key="join_db")
    
    # Collections selection
    collections_join = get_collections(selected_db_join)
    
    col1, col2 = st.columns(2)
    
    with col2:
        default_coll1_index = collections_join.index("Product") if "Product" in collections_join else 0
        collection1 = st.selectbox("Select Outer Collection:", collections_join, index=default_coll1_index, key="aggregate_coll1")
    
    with col1:
        remaining_collections = [c for c in collections_join if c != collection1]
        default_coll2_index = remaining_collections.index("Stock") if "Stock" in remaining_collections else 0
        collection2 = st.selectbox(
            "Select Inner Collection:", 
            remaining_collections,
            index=default_coll2_index,
            key="aggregate_coll2"
        )
    
    # Get properties for both collections
    properties1 = get_properties(selected_db_join, collection1)
    properties2 = get_properties(selected_db_join, collection2)
    
    # Configuration for Collection 2 (Inner Collection)
    st.subheader(f"Configuration for {collection2} (Inner Collection)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Default for Stock: ["IDW", "quantity"]
        default_output2 = [k for k in ["IDW", "quantity"] if k in properties2]
        if not default_output2:
            default_output2 = properties2[:min(2, len(properties2))]
        output_keys2 = st.multiselect(
            "Output Keys:",
            properties2,
            default=default_output2,
            key="aggregate_output2"
        )
    
    with col2:
        default_join2_index = properties2.index("IDP") if "IDP" in properties2 else 0
        join_key2 = st.selectbox(
            "Join Key:",
            properties2,
            index=default_join2_index,
            key="aggregate_join_key2"
        )
    
    with col3:
        # Stock has no filter keys in test.py
        filter_keys2 = st.multiselect(
            "Filter Keys:",
            properties2,
            default=[],
            key="aggregate_filter2"
        )
    
    with col4:
        group_by_key2 = st.selectbox(
            "Group By Key:",
            [""] + properties2,
            index=properties2.index("IDP")+1 if "IDP" in properties2 else 0,
            key="aggregate_groupby_key2"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        sharding2 = st.checkbox(f"Enable Sharding for {collection2}", value=True, key="aggregate_shard2")
    with col2:
        sharding_key2 = None
        if sharding2:
            default_shard2_index = properties2.index("IDP") if "IDP" in properties2 else 0
            sharding_key2 = st.selectbox(
                f"Sharding Key for {collection2}:",
                properties2,
                index=default_shard2_index,
                key="aggregate_shard_key2"
            )
    
    
        
    # Configuration for Collection 1 (Outer Collection)
    st.subheader(f"Configuration for {collection1} (Outer Collection)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Default for Product: ["name", "price"]
        default_output1 = [k for k in ["name", "price"] if k in properties1]
        if not default_output1:
            default_output1 = properties1[:min(2, len(properties1))]
        output_keys1 = st.multiselect(
            "Output Keys:",
            properties1,
            default=default_output1,
            key="aggregate_output1"
        )
    
    with col2:
        default_join1_index = properties1.index("IDP") if "IDP" in properties1 else 0
        join_key1 = st.selectbox(
            "Join Key:",
            properties1,
            index=default_join1_index,
            key="aggregate_join_key1"
        )
    
    with col3:
        default_filter1 = [k for k in ["brand"] if k in properties1]
        filter_keys1 = st.multiselect(
            "Filter Keys:",
            properties1,
            default=default_filter1,
            key="aggregate_filter1"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        sharding1 = st.checkbox(f"Enable Sharding for {collection1}", value=True, key="aggregate_shard1")
    with col2:
        sharding_key1 = None
        if sharding1:
            default_shard1_index = properties1.index("brand") if "brand" in properties1 else 0
            sharding_key1 = st.selectbox(
                f"Sharding Key for {collection1}:",
                properties1,
                index=default_shard1_index,
                key="aggregate_shard_key1"
            )
    st.divider()
    
    
    # add limit input
    limit = st.number_input("Limit (for final output):", min_value=1, value=100, step=1, key="aggregate_limit")
    
    # Compute button
    if st.button("Compute Aggregate Query Costs", type="primary", key="aggregate_compute"):
        if not output_keys1 and not output_keys2:
            st.error("Please select at least one output key for one collection")
        else:
            with st.spinner("Computing aggregate query costs..."):
                try:
                    results = compute_aggregate_query_costs(
                        database=selected_db_join,
                        collections=[collection1, collection2],
                        output_keys={collection1: output_keys1 or [], collection2: output_keys2 or []},
                        join_keys={collection1: join_key1, collection2: join_key2},
                        group_by_keys={collection1: None, collection2: group_by_key2 if group_by_key2 != "" else None},
                        filter_keys={collection1: filter_keys1 or [], collection2: filter_keys2 or []},
                        sharding={collection1: sharding1, collection2: sharding2},
                        sharding_keys={collection1: sharding_key1 if sharding1 else "", collection2: sharding_key2 if sharding2 else ""},
                        limit=limit
                    )
                    
                    # Display results
                    st.success("Aggregate query costs computed successfully!")
                    
                    # Per-collection metrics
                    col1, col2 = st.columns(2)
                    
                    with col2:
                        st.subheader(f"{collection1} Metrics (Outer Collection)")
                        st.metric("Servers Checked", f"{results['nb_servers_checked'][collection1]}")
                        st.metric("Output Documents", f"{results['nb_output_docs'][collection1]:,}")
                        st.metric("Scanned Doc Size", f"{results['scanned_doc_byte_size'][collection1]:,} B")
                        
                        scanned_size = results['scanned_data_byte_size'][collection1]
                        st.metric(
                            "Scanned Data",
                            f"{scanned_size:,} B" if scanned_size < 10**6 else f"{scanned_size/10**9:.3f} GB"
                        )
                        
                        output_size = results['output_data_byte_size'][collection1]
                        st.metric(
                            "Output Data",
                            f"{output_size:,} B" if output_size < 10**6 else f"{output_size/10**9:.3f} GB"
                        )
                        
                        shuffles = results.get('nb_shuffles', {}).get(collection1, 0)
                        st.metric("Number of Shuffles", f"{shuffles}")  
                                              
                        nb_loops = results.get("nb_loops", 0)
                        st.metric("Number of Loops", f"{nb_loops}")
    
                    
                    with col1:
                        st.subheader(f"{collection2} Metrics (Inner Collection)")
                        st.metric("Servers Checked", f"{results['nb_servers_checked'][collection2]}")
                        st.metric("Output Documents", f"{results['nb_output_docs'][collection2]:,}")
                        st.metric("Scanned Doc Size", f"{results['scanned_doc_byte_size'][collection2]:,} B")
                        
                        scanned_size = results['scanned_data_byte_size'][collection2]
                        st.metric(
                            "Scanned Data",
                            f"{scanned_size:,} B" if scanned_size < 10**6 else f"{scanned_size/10**9:.3f} GB"
                        )
                        
                        output_size = results['output_data_byte_size'][collection2]
                        st.metric(
                            "Output Data",
                            f"{output_size:,} B" if output_size < 10**6 else f"{output_size/10**9:.3f} GB"
                        )
                    
                        shuffles = results.get('nb_shuffles', {}).get(collection2, 0)
                        st.metric("Number of Shuffles", f"{shuffles}")


                    st.divider()
                    
                        
                    # Total costs
                    st.subheader("Total Query Costs")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Time Cost",
                            f"{results['time_cost_seconds']:.4f} s"
                        )
                    
                    with col2:
                        st.metric(
                            "Carbon Footprint",
                            f"{results['carbon_footprint_kgCO2eq']:.6f} kgCO2eq"
                        )
                    
                    with col3:
                        st.metric(
                            "Price Cost",
                            f"{results['price_cost_€']:.8f} €"
                        )
                    
                    # Show raw results in expander
                    with st.expander("View Raw Results"):
                        st.json(results)
                        
                except Exception as e:
                    st.error(f"Error computing query costs: {str(e)}")