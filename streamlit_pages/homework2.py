import streamlit as st
from pathlib import Path

from filter import compute_filter_query_costs, compute_join_query_costs
from loader import load_schemas_from_folder
from size import get_all_properties
from settings import NB_DOCS

st.set_page_config(page_title="Homework 2", layout="wide")

st.title("Homework 2")

tab1, tab2 = st.tabs(["Filter Query Costs", "Join Query Costs"])

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

# Filter Query Costs Tab
with tab1:
    st.header("Filter Query Costs")
    st.write("Compute the costs of executing a filter query on a collection")
    
    # Database selection
    databases = get_available_databases()
    default_db_index = databases.index("db1") if "db1" in databases else 0
    selected_db = st.selectbox("Select Database:", databases, index=default_db_index, key="filter_db")
    
    # Collection selection
    collections = get_collections(selected_db)
    default_coll_index = collections.index("Stock") if "Stock" in collections else 0
    selected_collection = st.selectbox("Select Collection:", collections, index=default_coll_index, key="filter_collection")
    
    # Get available properties
    properties = get_properties(selected_db, selected_collection)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Output keys selection
        st.subheader("Output Keys")
        default_output = [k for k in ["quantity", "location"] if k in properties]
        if not default_output:
            default_output = properties[:min(2, len(properties))]
        output_keys = st.multiselect(
            "Select keys to include in output:",
            properties,
            default=default_output,
            key="filter_output"
        )
    
    with col2:
        # Filter keys selection
        st.subheader("Filter Keys")
        default_filter = [k for k in ["IDP", "IDW"] if k in properties]
        filter_keys = st.multiselect(
            "Select keys to filter by:",
            properties,
            default=default_filter,
            key="filter_keys"
        )
    
    # Sharding options
    st.subheader("Sharding Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        sharding_enabled = st.checkbox("Enable Sharding", value=True, key="filter_sharding")
    
    with col2:
        sharding_key = None
        if sharding_enabled:
            default_shard_index = properties.index("IDP") if "IDP" in properties else 0
            sharding_key = st.selectbox(
                "Sharding Key:",
                properties,
                index=default_shard_index,
                key="filter_sharding_key"
            )
    
    # Compute button
    if st.button("Compute Filter Query Costs", type="primary", key="filter_compute"):
        if not output_keys:
            st.error("Please select at least one output key")
        else:
            with st.spinner("Computing query costs..."):
                try:
                    results = compute_filter_query_costs(
                        database=selected_db,
                        collection=selected_collection,
                        output_keys=output_keys,
                        filter_keys=filter_keys,
                        sharding=sharding_enabled,
                        sharding_key=sharding_key if sharding_enabled else None
                    )
                    
                    # Display results
                    st.success("Query costs computed successfully!")
                    
                    # Server and document metrics
                    st.subheader("Query Execution Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Servers Checked",
                            f"{results['nb_servers_checked']}"
                        )
                    
                    with col2:
                        st.metric(
                            "Output Documents",
                            f"{results['nb_output_docs']:,}"
                        )
                    
                    with col3:
                        st.metric(
                            "Scanned Doc Size",
                            f"{results['scanned_doc_byte_size']:,} B"
                        )
                    
                    # Data size metrics
                    st.subheader("Data Size Metrics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        scanned_size = results['scanned_data_byte_size']
                        st.metric(
                            "Total Scanned Data",
                            f"{scanned_size:,} B" if scanned_size < 10**6 else f"{scanned_size/10**9:.3f} GB"
                        )
                    
                    with col2:
                        output_size = results['output_data_byte_size']
                        st.metric(
                            "Total Output Data",
                            f"{output_size:,} B" if output_size < 10**6 else f"{output_size/10**9:.3f} GB"
                        )
                    
                    # Cost metrics
                    st.subheader("Query Costs")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Time Cost",
                            f"{results['time_cost_seconds']:.4f} s"
                        )
                    
                    with col2:
                        st.metric(
                            "Carbon Footprint",
                            f"{results['carbon_footprint_gCO2']:.6f} gCO2"
                        )
                    
                    with col3:
                        st.metric(
                            "Price Cost",
                            f"${results['price_cost_$']:.8f}"
                        )
                    
                    # Show raw results in expander
                    with st.expander("View Raw Results"):
                        st.json(results)
                        
                except Exception as e:
                    st.error(f"Error computing query costs: {str(e)}")

# Join Query Costs Tab
with tab2:
    st.header("Join Query Costs")
    st.write("Compute the costs of executing a join query between two collections")
    
    # Database selection
    databases_join = get_available_databases()
    default_db_join_index = databases_join.index("db1") if "db1" in databases_join else 0
    selected_db_join = st.selectbox("Select Database:", databases_join, index=default_db_join_index, key="join_db")
    
    # Collections selection
    collections_join = get_collections(selected_db_join)
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_coll1_index = collections_join.index("Product") if "Product" in collections_join else 0
        collection1 = st.selectbox("Select First Collection:", collections_join, index=default_coll1_index, key="join_coll1")
    
    with col2:
        remaining_collections = [c for c in collections_join if c != collection1]
        default_coll2_index = remaining_collections.index("Stock") if "Stock" in remaining_collections else 0
        collection2 = st.selectbox(
            "Select Second Collection:", 
            remaining_collections,
            index=default_coll2_index,
            key="join_coll2"
        )
    
    # Get properties for both collections
    properties1 = get_properties(selected_db_join, collection1)
    properties2 = get_properties(selected_db_join, collection2)
    
    # Configuration for Collection 1
    st.subheader(f"Configuration for {collection1}")
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
            key="join_output1"
        )
    
    with col2:
        default_join1_index = properties1.index("IDP") if "IDP" in properties1 else 0
        join_key1 = st.selectbox(
            "Join Key:",
            properties1,
            index=default_join1_index,
            key="join_key1"
        )
    
    with col3:
        default_filter1 = [k for k in ["brand"] if k in properties1]
        filter_keys1 = st.multiselect(
            "Filter Keys:",
            properties1,
            default=default_filter1,
            key="join_filter1"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        sharding1 = st.checkbox(f"Enable Sharding for {collection1}", value=True, key="join_shard1")
    with col2:
        sharding_key1 = None
        if sharding1:
            default_shard1_index = properties1.index("brand") if "brand" in properties1 else 0
            sharding_key1 = st.selectbox(
                f"Sharding Key for {collection1}:",
                properties1,
                index=default_shard1_index,
                key="join_shard_key1"
            )
    
    st.divider()
    
    # Configuration for Collection 2
    st.subheader(f"Configuration for {collection2}")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Default for Stock: ["IDW", "quantity"]
        default_output2 = [k for k in ["IDW", "quantity"] if k in properties2]
        if not default_output2:
            default_output2 = properties2[:min(2, len(properties2))]
        output_keys2 = st.multiselect(
            "Output Keys:",
            properties2,
            default=default_output2,
            key="join_output2"
        )
    
    with col2:
        default_join2_index = properties2.index("IDP") if "IDP" in properties2 else 0
        join_key2 = st.selectbox(
            "Join Key:",
            properties2,
            index=default_join2_index,
            key="join_key2"
        )
    
    with col3:
        # Stock has no filter keys in test.py
        filter_keys2 = st.multiselect(
            "Filter Keys:",
            properties2,
            default=[],
            key="join_filter2"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        sharding2 = st.checkbox(f"Enable Sharding for {collection2}", value=True, key="join_shard2")
    with col2:
        sharding_key2 = None
        if sharding2:
            default_shard2_index = properties2.index("IDP") if "IDP" in properties2 else 0
            sharding_key2 = st.selectbox(
                f"Sharding Key for {collection2}:",
                properties2,
                index=default_shard2_index,
                key="join_shard_key2"
            )
    
    # Compute button
    if st.button("Compute Join Query Costs", type="primary", key="join_compute"):
        if not output_keys1 or not output_keys2:
            st.error("Please select at least one output key for each collection")
        else:
            with st.spinner("Computing join query costs..."):
                try:
                    results = compute_join_query_costs(
                        database=selected_db_join,
                        collections=[collection1, collection2],
                        output_keys={collection1: output_keys1, collection2: output_keys2},
                        join_keys={collection1: join_key1, collection2: join_key2},
                        filter_keys={collection1: filter_keys1, collection2: filter_keys2},
                        sharding={collection1: sharding1, collection2: sharding2},
                        sharding_keys={collection1: sharding_key1 if sharding1 else "", collection2: sharding_key2 if sharding2 else ""}
                    )
                    
                    # Display results
                    st.success("Join query costs computed successfully!")
                    
                    # Per-collection metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"{collection1} Metrics")
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
                    
                    with col2:
                        st.subheader(f"{collection2} Metrics")
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
                    
                    # Total costs
                    st.subheader("Total Join Query Costs")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Time Cost",
                            f"{results['time_cost_seconds']:.4f} s"
                        )
                    
                    with col2:
                        st.metric(
                            "Carbon Footprint",
                            f"{results['carbon_footprint_gCO2']:.6f} gCO2"
                        )
                    
                    with col3:
                        st.metric(
                            "Price Cost",
                            f"${results['price_cost_$']:.8f}"
                        )
                    
                    # Show raw results in expander
                    with st.expander("View Raw Results"):
                        st.json(results)
                        
                except Exception as e:
                    st.error(f"Error computing join query costs: {str(e)}")