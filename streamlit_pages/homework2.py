import streamlit as st
import json
from pathlib import Path
from size import compute_db_size
from sharding import compute_sharding_distribution
from settings import NB_DOCS

st.set_page_config(page_title="Homework 2", layout="wide")

st.title("Homework 2: Database Analysis")