import streamlit as st
from pathlib import Path

from size import compute_db_size
from sharding import compute_sharding_distribution
from settings import NB_DOCS

st.set_page_config(page_title="Homework 1", layout="wide")

st.title("Homework 1")

tab1, tab2 = st.tabs(["Database Size Computation", "Sharding Distribution"])

def get_schema(db_name):
    if db_name == "db1":
        return "DB1 - Prod{[Cat],Supp}, St, Wa, OL, Cl"
    elif db_name == "db2":
        return "DB2 - Prod{[Cat],Supp, [St]}, Wa, OL, Cl"
    elif db_name == "db3":
        return "DB3 - St{Prod{[Cat],Supp}}, Wa, OL, Cl"
    elif db_name == "db4":
        return "DB4 - St, Wa, OL{Prod{[Cat],Supp}}, Cl"
    elif db_name == "db5":
        return "DB5 - Prod{[Cat],Supp, [OL]}, St, Wa, Cl"

# Database Size Computation Tab
with tab1:
    st.header("Database Size Computation")
    st.write("Calculate the estimated size of different database schemas")
    
    # Get available schemas
    schemas_path = Path("schemas")
    available_schemas = [d.name for d in schemas_path.iterdir() if d.is_dir() and d.name != "default"]
    
    selected_schema = st.selectbox(
        "Select Database Schema:",
        available_schemas,
        format_func=get_schema
    )
    
    # Automatically compute database size
    with st.spinner("Computing database size..."):
        results = compute_db_size(f"schemas/{selected_schema}")
        
        # Display results in a structured way
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Document Sizes")
            for collection, sizes in results.items():
                if collection != "total_database_byte_size" and isinstance(sizes, dict):
                    doc_size = sizes.get("document_byte_size", 0)
                    st.metric(f"{collection}", f"{doc_size:,} B")
        
        with col2:
            st.subheader("Collection Sizes")
            for collection, sizes in results.items():
                if collection != "total_database_byte_size" and isinstance(sizes, dict):
                    coll_size = sizes.get("collection_byte_size", 0)
                    if coll_size > 10**6:
                        st.metric(f"{collection}", f"{coll_size/10**9:.3f} GB")
                    else:
                        st.metric(f"{collection}", f"{coll_size} B")
        
        total_size = results.get("total_database_byte_size", 0)
        st.subheader(f"Total Database Size : {total_size/10**9:.3f} GB")

# Sharding Distribution Tab
with tab2:
    st.header("Sharding Distribution")
    st.write("Analyze how data would be distributed across servers with different sharding strategies")
    
    # Get available collections
    collections = list(NB_DOCS.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_collection = st.selectbox(
            "Select Collection:",
            collections,
            index=collections.index("Stock") if "Stock" in collections else 0
        )
    
    with col2:
        sharding_key = st.text_input(
            "Sharding Key:",
            value="IDP",
            placeholder="Enter the sharding key (e.g., IDP, IDW, etc.)"
        )
    
    if st.button("Compute Sharding Distribution", type="primary"):
        with st.spinner("Computing sharding distribution..."):
            try:
                results = compute_sharding_distribution(selected_collection, sharding_key)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Documents per Server",
                    f"{results['docs_per_server']}"
                )
            
            with col2:
                st.metric(
                    "Distinct Values per Server",
                    f"{results['distinct_values_per_server']}"
                )
            
            with col3:
                st.metric(
                    "Number of Servers Used",
                    f"{results['nb_servers_used']}"
                )