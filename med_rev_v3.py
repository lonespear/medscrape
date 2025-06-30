import streamlit as st
from streamlit_tags import st_tags
from Bio import Entrez

import pandas as pd
import numpy as np

# Plotly imports for interactive visualization
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
from umap import UMAP

# Set your email for Entrez
Entrez.email = "jonathan.day@westpoint.edu"

#############################
#  PubMed Query Functions   #
#############################

def search_pubmed(query, max_results=2000):
    """
    Search PubMed for a given query and return the XML records.
    """
    # Use the esearch utility to search PubMed
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    id_list = record["IdList"]

    # Use the efetch utility to fetch details for each PMID
    handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
    records = Entrez.read(handle)
    handle.close()
    
    return records

def parse_articles(records, long_study=True, rev_paper=True, sys_rev=True, 
                   clinical_trial=True, meta_analysis=True, rand_ct=True):
    """
    Parse PubMed XML records into a list of dictionaries.
    """
    data = []
    for article in records["PubmedArticle"]:
        # Extract authors
        if "AuthorList" in article["MedlineCitation"]["Article"]:
            authors_list = []
            for author in article["MedlineCitation"]["Article"]["AuthorList"]:
                last_name = author.get("LastName", "")
                fore_name = author.get("ForeName", "")
                authors_list.append(last_name + " " + fore_name)
            authors = ", ".join(authors_list)
        else:
            authors = ""
        
        title = article["MedlineCitation"]["Article"]["ArticleTitle"]
        # Get abstract text (or an empty string if not available)
        abstract = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [""])[0]
        pubmed_id = article["MedlineCitation"]["PMID"]
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
        pub_year = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"].get("Year", "N/A")
        journal_name = article["MedlineCitation"]["Article"]["Journal"]["Title"]

        # Flags for publication types
        article_type_list = article["MedlineCitation"]["Article"].get("PublicationTypeList", [])
        is_review = any(pub_type.lower() == "review" for pub_type in article_type_list)
        is_sys_rev = any(pub_type.lower() == "systematic review" for pub_type in article_type_list)
        is_clinical_trial = any(pub_type.lower() == "clinical trial" for pub_type in article_type_list)
        is_meta_analysis = any(pub_type.lower() == "meta-analysis" for pub_type in article_type_list)
        is_randomized_controlled_trial = any(pub_type.lower() == "randomized controlled trial" for pub_type in article_type_list)
        longitudinal_terms = ["longitudinal", "long-term follow up", "long term follow up", "follow-up", "follow up"]
        longitudinal_study = any(term in abstract.lower() for term in longitudinal_terms)

        article_data = {
            "PublicationYear": pub_year,
            "Authors": authors,
            "Title": title,
            "Abstract": abstract,
            "JournalName": journal_name,
            "PubMedURL": url,
            "Review": 1 if is_review else 0,
            "SysRev": 1 if is_sys_rev else 0,
            "ClinicalTrial": 1 if is_clinical_trial else 0,
            "MetaAnalysis": 1 if is_meta_analysis else 0,
            "RCT": 1 if is_randomized_controlled_trial else 0,
            "LongitudinalStudy": 1 if longitudinal_study else 0
        }
        data.append(article_data)
    
    # Remove keys if the option is unchecked
    if not long_study:
        for row in data:
            row.pop("LongitudinalStudy", None)
    if not rev_paper:
        for row in data:
            row.pop("Review", None)
    if not sys_rev:
        for row in data:
            row.pop("SysRev", None)
    if not clinical_trial:
        for row in data:
            row.pop("ClinicalTrial", None)
    if not meta_analysis:
        for row in data:
            row.pop("MetaAnalysis", None)
    if not rand_ct:
        for row in data:
            row.pop("RCT", None)
    
    return data

#############################
#  Clustering Functions     #
#############################

def run_cluster(df, n_clusters, n_key_words, dimred_method, cluster_method):
    """
    Perform dimensionality reduction and clustering on the DataFrame.
    Returns:
        cluster_sizes: Series with counts for each cluster.
        cluster_keywords: Dict with top keywords per cluster.
        centroids: Array of centroids if using K-Means; else None.
    """
    # Define dimensionality reduction methods
    dim_reduction_methods = {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, random_state=42),
        "UMAP": UMAP(n_components=2, random_state=42)
    }
    # Define clustering algorithms
    clustering_methods = {
        "K-Means": KMeans(n_clusters=n_clusters, random_state=42),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters),
        "LDA": LatentDirichletAllocation(n_components=n_clusters, random_state=42)
    }
    
    # Create a corpus by concatenating Title and Abstract
    text_data = df['Title'].fillna('') + ' ' + df['Abstract'].fillna('')
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_data)
    
    # Dimensionality reduction
    dim_reducer = dim_reduction_methods.get(dimred_method)
    reduced_data = dim_reducer.fit_transform(tfidf_matrix.toarray())
    df[['Dim_1', 'Dim_2']] = reduced_data[:, :2]
    
    # Clustering
    cluster_model = clustering_methods.get(cluster_method)
    if cluster_method == "LDA":
        cluster_model.fit(tfidf_matrix)
        cluster_labels = cluster_model.transform(tfidf_matrix).argmax(axis=1)
        centroids = None
    else:
        cluster_labels = cluster_model.fit_predict(tfidf_matrix)
        if cluster_method == "K-Means":
            centroids = cluster_model.cluster_centers_
        else:
            centroids = None
    df['Cluster'] = cluster_labels
    cluster_sizes = df['Cluster'].value_counts()

    # Extract top keywords for each cluster
    cluster_keywords = {}
    terms = vectorizer.get_feature_names_out()
    if cluster_method == "K-Means":
        order_centroids = cluster_model.cluster_centers_.argsort()[:, ::-1]
        for i in range(n_clusters):
            cluster_keywords[i] = [terms[ind] for ind in order_centroids[i, :n_key_words]]
    elif cluster_method == "LDA":
        for i, topic in enumerate(cluster_model.components_):
            cluster_keywords[i] = [terms[ind] for ind in topic.argsort()[:-n_key_words - 1:-1]]
    else:  # For Hierarchical and DBSCAN
        for cluster in range(n_clusters):
            cluster_indices = df[df['Cluster'] == cluster].index
            if len(cluster_indices) > 0:
                cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
                # For sparse matrix, convert to array and then sort:
                top_indices = np.array(cluster_tfidf).argsort()[0, -n_key_words:][::-1]
                cluster_keywords[cluster] = [terms[ind] for ind in top_indices]
            else:
                cluster_keywords[cluster] = []
    
    return cluster_sizes, cluster_keywords, centroids

def plot_dimred_interactive(df, dimred_method, cluster_method, centroids=None):
    """
    Create an interactive Plotly scatter plot of the dimensionality-reduced data with clusters.
    Hovering over a point shows the abstract (along with Title and JournalName).
    If centroids are provided (K-Means), plot them as well.
    """
    # Create the interactive scatter plot using Plotly Express
    fig = px.scatter(
        df,
        x="Dim_1",
        y="Dim_2",
        color="Cluster",
        hover_data={"Abstract": True, "Title": True, "JournalName": True},
        title=f"{dimred_method} + {cluster_method}",
        labels={"Dim_1": "Dimension 1", "Dim_2": "Dimension 2"}
    )
    
    # Add centroids for K-Means
    if centroids is not None:
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers",
                marker=dict(symbol="x", size=12, color="red"),
                name="Centroids",
                hoverinfo="skip"
            )
        )
    # Enable zoom and interactive hover
    fig.update_layout(hovermode="closest")
    return fig

#############################
#  Main App Interface       #
#############################

# Default search terms for tagging
adh_terms = [
    "Atypical ductal hyperplasia", 
    "Atypical hyperplasia of the breast", 
    "Atypical breast hyperplasia", 
    "Ductal hyperplasia with atypia", 
    "Atypical proliferation of ductal cells", 
    "Premalignant breast lesion", 
    "Atypical epithelial hyperplasia", 
    "Breast atypia", 
    "Proliferative breast lesion with atypia", 
    "Atypical intraductal hyperplasia", 
    "ADH (atypical ductal hyperplasia)", 
    "ADL", 
    "ADH"
]

breast_cancer_terms = [
    "LCIS", "Lobular carcinoma in situ", "Breast cancer",
    "breast cancer", "mammary carcinoma", "invasive ductal carcinoma (IDC)", 
    "invasive lobular carcinoma (ILC)", "ductal carcinoma in situ (DCIS)", 
    "lobular carcinoma in situ (LCIS)", "triple-negative breast cancer",
    "HER2-positive breast cancer", "BRCA1 mutations", "BRCA2 mutations", 
    "metastatic breast cancer", "hormone receptor-positive breast cancer", 
    "estrogen receptor-positive (ER-positive)", "progesterone receptor-positive (PR-positive)"
]

# Initialize session state for the DataFrame if not already present
if 'df' not in st.session_state:
    st.session_state.df = None

st.set_page_config(layout="wide")

# App Title and Description
st.title("Systematic Review Tool for PubMed")
st.write("""
         **This tool aids medical researchers in conducting comprehensive literature reviews by performing broad queries on 
         PubMed to capture a vast array of articles. Utilizing NLP techniques, it systematically analyzes the retrieved 
         articles (parsing details such as abstracts, authors, and publication types) and applies clustering algorithms to 
         group articles by thematic similarities. This structured approach enhances the efficiency and accuracy of literature reviews.**
         """)
st.divider()

# CSV Upload Section
uploaded_file = st.file_uploader("Drag and drop or select a CSV file", type=["csv"])
if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.session_state.df = uploaded_df
    if 'Abstract' in uploaded_df.columns:
        st.success("CSV uploaded successfully and 'Abstract' column detected.")
    else:
        st.error("The uploaded CSV does not contain an 'Abstract' column. Please upload a valid dataset.")

st.divider()
st.write('OR')

# Scraping Tool Section
st.subheader("Scraping Tool")
included_1 = st_tags(
    value=adh_terms,
    label="#### Include words or phrases:",
    text="Press enter to add more",
    key='included1'
)
included_2 = st_tags(
    value=breast_cancer_terms,
    label="#### Include words or phrases:",
    text="Press enter to add more",
    key='included2'
)
excluded = st_tags(
    label="##### Exclude words or phrases:",
    text="Press enter to add more",
    maxtags=4,
    key='excluded'
)

# Build query string
query_inc1 = " OR ".join([f'"{term}"' for term in included_1])
query_inc2 = " OR ".join([f'"{term}"' for term in included_2])
query_ex = " AND ".join([f'NOT "{term}"' for term in excluded])
query = f'({query_inc1} AND {query_inc2}) {query_ex}'

st.write('##### Final Query')
st.write(query)

# Query Options Section
c3, c4, c5, c6, _, c9 = st.columns([3, 4, 4, 4, 8, 3])
with c3:
    max_results = st.number_input('Maximum Results', min_value=500, max_value=20000, value=2000, step=1)
with c4:
    long_study = st.checkbox("Longitudinal Study", value=True, key='long')
    rev_paper = st.checkbox("Review Paper", value=True, key='rev')
with c5:
    sys_rev = st.checkbox("Systematic Review", value=True, key='sys')
    clin_trial = st.checkbox("Clinical Trial", value=True, key='clin')
with c6:
    meta = st.checkbox("Meta Analysis", value=True, key='meta')
    rct = st.checkbox("RCT", value=True, key='rct')
with c9:
    run_query = st.button("Run Search", type='primary')

# Run PubMed Search
if run_query:
    if not included_1:
        st.error("Query is empty!")
    else:
        placeholder = st.empty()
        placeholder.write("Searching PubMed...")
        # Replace with your actual email if needed
        Entrez.email = "your.email@example.com"
        records = search_pubmed(query, max_results=max_results)
        articles = parse_articles(records, long_study=long_study, rev_paper=rev_paper, 
                                  sys_rev=sys_rev, clinical_trial=clin_trial, 
                                  meta_analysis=meta, rand_ct=rct)
        st.session_state.df = pd.DataFrame(articles)
        placeholder.empty()

# Display Search Results Section
if st.session_state.df is not None and not st.session_state.df.empty:
    st.subheader("Search Results")
    num_results = len(st.session_state.df.index)
    st.write(st.session_state.df, f'Total Number of Results: {num_results}')
    csv = st.session_state.df.to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name='pubmed_search_results.csv',
        mime='text/csv'
    )

st.divider()

# Clustering Analysis Section
st.subheader("Clustering Analysis")
c10, c11, c12, c13, _, c19 = st.columns([5, 5, 4, 4, 8, 3])
with c10:
    cluster_method = st.selectbox('Select Clustering Algorithm', 
                                  ("K-Means", "LDA", "DBSCAN", "Hierarchical"), index=0)
with c11:
    dimred_method = st.selectbox('Select Dimensionality Reduction', 
                                 ("PCA", "t-SNE", "UMAP"), index=0)
with c12:
    num_clusters = st.number_input("Number of Clusters", 2, 10, 5, 1)
with c13:
    n_keywords = st.number_input("Number of Keywords", 1, 20, 10, 1)
with c19:
    cluster_btn = st.button("Begin Cluster", type='primary')

if cluster_btn:
    if st.session_state.df is None or st.session_state.df.empty:
        st.error("No data to cluster!")
    else:
        cluster_sizes, cluster_keywords, centroids = run_cluster(
            st.session_state.df, n_clusters=num_clusters, n_key_words=n_keywords,
            dimred_method=dimred_method, cluster_method=cluster_method
        )
        # Generate the interactive plot using Plotly
        fig = plot_dimred_interactive(st.session_state.df, dimred_method, cluster_method, centroids)
        st.write("### Updated Search Results with Clusters", st.session_state.df)
        csv_clustered = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="Download clustered results as CSV",
            data=csv_clustered,
            file_name='pubmed_clustered_results.csv',
            mime='text/csv'
        )
        # Use two columns to display cluster sizes and keywords
        c20, c21 = st.columns(2)
        with c20:
            st.write("Cluster Sizes:", cluster_sizes)
        with c21:
            st.write(pd.DataFrame.from_dict(cluster_keywords, orient='index', 
                      columns=[f'Term {i+1}' for i in range(n_keywords)]))
        # Display the interactive Plotly chart
        st.plotly_chart(fig, use_container_width=True)
