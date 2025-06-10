import streamlit as st
import pandas as pd

# --------------------------------------------------
# Data Structures & Data Processing in Python Cheatsheet
# --------------------------------------------------

# Sidebar for page navigation
page = st.sidebar.radio("Navigate to Section", ["Home", "File Inspection", "Data Conversion", "Data Quality Check", "Data Validation"])

# Page content
if page == "Home":
    st.title("Data Processing in Python: A Quick Reference")
    st.markdown("Commonly used Python codes for general data preparation, cleaning, and profiling.")

    st.set_page_config(
        page_title="Data Processing Python Cheatsheet",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Styled Page Title
    st.markdown(
        "<h1 style='text-align:center; color:#4B8BBE;'>📘 Data Processing Python Cheatsheet</h1>",
        unsafe_allow_html=True
    )

    # Data: each section from lecture slides
    cheat_data = [
        {
            "Topic": "Recursion: Factorial",
            "Summary": "Compute n! via self-call with base case 0! = 1",
            "Code": '''
    # Factorial Function (DSA2.pdf)

    def fact(n):
        if n == 0:
            return 1
        else:
            return n * fact(n-1)

    n = int(input("Enter n: "))
    print(f"factorial({n}) = {fact(n)}")
    ''',
            "Explanation": "Recursive factorial with base case n=0. O(n) time, O(n) space."
        },
    ]       

elif page == "File Inspection":
    st.title("File Inspection")
    st.markdown("This page is for inspecting files and reading them as data frames.")
    st.markdown(""" #### Detect File Extensions """)
    code = '''
    def greet(name):
        return f"Hello, {name}!"
    '''
    st.code(code, language='python')
elif page == "Data Conversion":
    st.title("Data Conversion")
    st.markdown("This page is for converting different file types to CSV file.")
elif page == "Data Quality Check":
    st.title("Data Quality Check")
    st.markdown("This page is for performing data quality checks and identifying duplicates.")
elif page == "Data Validation":
    st.title("Data Validation")
    st.markdown("This page is for validating data such as phone numbers, ids and email addresses.")    

# Add a footer (optional)
st.markdown("---")
st.markdown("© 2025 My Company")

# Sidebar: simple search bar and topic list
st.sidebar.header("🔎 Search Topics")
search_query = st.sidebar.text_input("Enter keyword to filter topics:")

# Display all or filtered topics in sidebar
st.sidebar.markdown("### Topics")
for item in cheat_data:
    if not search_query or search_query.lower() in item['Topic'].lower():
        st.sidebar.write(f"- {item['Topic']}")

# Optional summary table toggle
if st.sidebar.checkbox("Show Summary Table"):
    df_summary = pd.DataFrame([
        {"Topic": t['Topic'], "Summary": t['Summary']}
        for t in cheat_data
        if not search_query or search_query.lower() in t['Topic'].lower()
    ])
    st.sidebar.dataframe(df_summary, use_container_width=True)

# Expand All control
expand_all = st.sidebar.button("Expand All")

# Filter main content based on search
filtered = [
    t for t in cheat_data
    if not search_query or search_query.lower() in t['Topic'].lower()
]

# Main Content
for item in filtered:
    with st.expander(f"🔹 {item['Topic']}", expanded=expand_all):
        st.markdown(f"**Summary**: {item['Summary']}")
        if code := item.get("Code"):
            st.code(code, language='python')
        st.markdown(f"**Explanation**: {item['Explanation']}")
        st.markdown("---")

# Download Summary CSV
if filtered:
    df_download = pd.DataFrame([
        {"Topic": t['Topic'], "Summary": t['Summary']} for t in filtered
    ])
    csv_data = df_download.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Summary CSV",
        data=csv_data,
        file_name="dataprocessing_python_cheatsheet.csv",
        mime="text/csv"
    )
else:
    st.info("No topics match your search.")