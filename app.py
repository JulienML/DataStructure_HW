import streamlit as st

st.title("Big Data Structure - Homeworks")
st.text("Julien DE VOS - Lorrain MORLET - Aymeric MARTIN")

homework1 = st.Page("./streamlit_pages/homework1.py", title="Homework 1", default=True)
homework2 = st.Page("./streamlit_pages/homework2.py", title="Homework 2")

pg = st.navigation(
    [
        homework1,
        homework2
    ]
)
pg.run()