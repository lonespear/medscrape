import streamlit as st
from streamlit_tags import st_tags
from Bio import Entrez

import pandas as pd
import requests

# Plotly imports for interactive visualization
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
from umap import UMAP

# Set your email for Entrez and token for huggingface
Entrez.email = "jonathan.day@westpoint.edu"
HF_TOKEN = st.secrets["hf_token"]

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
#  Clustering and Summarization Functions     #
#############################

def run_cluster(df, n_clusters, n_key_words, dimred_method, cluster_method):
    """
    Perform dimensionality reduction and clustering on the DataFrame.
    Returns:
        cluster_sizes: Series with counts for each cluster (sorted by size descending).
        cluster_keywords: Dict with unique top keywords per cluster.
        cluster_keyword_scores: Dict with scores associated with keywords.
        centroids: Array of centroids if using K-Means; else None.
    """
    from collections import defaultdict

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

    # Create text corpus
    text_data = df['Title'].fillna('') + ' ' + df['Abstract'].fillna('')
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_data)
    terms = vectorizer.get_feature_names_out()
    
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
        centroids = cluster_model.cluster_centers_ if cluster_method == "K-Means" else None

    df['Cluster'] = cluster_labels
    cluster_sizes = df['Cluster'].value_counts().sort_values(ascending=False)  # Sort by size

    # Initialize output dicts
    cluster_keywords = {}
    cluster_keyword_scores = {}
    used_keywords = set()

    # Iterate over clusters by size
    for cluster in cluster_sizes.index:
        keywords = []
        scores = []

        if cluster_method == "K-Means":
            centroid_weights = cluster_model.cluster_centers_[cluster]
            keyword_score_pairs = sorted(
                [(terms[i], centroid_weights[i]) for i in range(len(terms))],
                key=lambda x: x[1],
                reverse=True
            )

        elif cluster_method == "LDA":
            topic_weights = cluster_model.components_[cluster]
            keyword_score_pairs = sorted(
                [(terms[i], topic_weights[i]) for i in range(len(terms))],
                key=lambda x: x[1],
                reverse=True
            )

        else:  # Hierarchical / DBSCAN — use mean TF-IDF scores
            cluster_indices = df[df["Cluster"] == cluster].index
            if len(cluster_indices) == 0:
                cluster_keywords[cluster] = []
                cluster_keyword_scores[cluster] = []
                continue
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
            keyword_score_pairs = sorted(
                [(terms[i], cluster_tfidf[0, i]) for i in range(len(terms))],
                key=lambda x: x[1],
                reverse=True
            )

        # Assign top n_key_words that are not already used
        for kw, score in keyword_score_pairs:
            if kw not in used_keywords:
                keywords.append(kw)
                scores.append(score)
                used_keywords.add(kw)
            if len(keywords) == n_key_words:
                break

        cluster_keywords[cluster] = keywords
        cluster_keyword_scores[cluster] = scores

    return cluster_sizes, cluster_keywords, cluster_keyword_scores, centroids

def plot_dimred_interactive(df, dimred_method, cluster_method, centroids=None):
    """
    Create an interactive Plotly scatter plot with dimensionality-reduced data and clusters.
    """
    df["Cluster"] = df["Cluster"].astype(str)  # Treat clusters as categories

    fig = px.scatter(
        df,
        x="Dim_1",
        y="Dim_2",
        color="Cluster",
        hover_data={
            "Citation": True,
            "Dim_1": False,
            "Dim_2": False,
            "Cluster": False
        },
        title=f"{dimred_method} + {cluster_method} Clustering Visualization",
        labels={"Dim_1": "Dimension 1", "Dim_2": "Dimension 2"},
        color_discrete_sequence=px.colors.qualitative.Set1,
        height=700
    )
    
    if centroids is not None:
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers+text",
                marker=dict(symbol="x", size=12, color="red"),
                text=[f"C{i}" for i in range(len(centroids))],
                textposition="top center",
                name="Centroids",
                hoverinfo="skip"
            )
        )

    fig.update_layout(hovermode="closest")
    return fig

def summarize_huggingface_api(text, num_sentences=3, hf_token=None):
    """
    Summarize using Hugging Face Inference API (e.g., `facebook/bart-large-cnn`)
    """
    if hf_token is None:
        hf_token = st.secrets["hf_token"]

    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": text, "parameters": {"min_length": 25, "max_length": 100}}

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        st.error(f"Summarization API failed: {response.text}")
        return "Summarization failed."

    summary = response.json()[0]["summary_text"]
    return summary


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

st.markdown("""
### About This App

This tool facilitates comprehensive literature reviews using PubMed data. It allows users to:
- Query a large volume of biomedical articles using customizable keyword filters.
- Automatically parse abstracts and metadata.
- Apply clustering and dimensionality reduction to explore thematic groupings.
- Summarize each cluster's content using natural language models.

The aim is to support researchers in identifying relevant trends, topics, and insights across vast bodies of academic literature.
""")

st.divider()

st.subheader("Data Scraping Tool")
source_option = st.radio("Use criteria below to scrape PubMed for journal articles or upload your own CSV using the format described.", options=["Scrape from PubMed", "Upload CSV"])


# CSV Upload Section
if source_option == "Upload CSV":
    st.markdown("""
        **📄 Required CSV Columns:**

        To ensure full functionality, your uploaded CSV must include at least the following columns:

        - `Title`: The title of the article.
        - `Abstract`: The abstract of the article.
        - `Authors`: The list of authors.
        - `PublicationYear`: Year the article was published.
        - `JournalName`: Name of the journal.
        - `PubMedURL`: URL to the article on PubMed (optional but recommended).

        **Example:**

        | Title | Abstract | Authors | PublicationYear | JournalName | PubMedURL |
        |-------|----------|---------|------------------|--------------|-----------|
        | The Role of ADH in Breast Cancer | This study explores... | Smith J, Lee A | 2022 | JAMA Oncology | https://... |
        """)
    uploaded_file = st.file_uploader("Drag and drop or select a CSV file", type=["csv"])
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.session_state.df = uploaded_df
        if 'Abstract' in uploaded_df.columns:
            st.success("CSV uploaded successfully and 'Abstract' column detected.")
        else:
            st.error("The uploaded CSV does not contain an 'Abstract' column. Please upload a valid dataset.")

# Scrape Data Option
elif source_option == "Scrape from PubMed":

    st.markdown("""
    ### Constructing a PubMed Search

    You may enter keywords and phrases in the boxes below:
    - **Include (Group 1 and 2):** Articles must contain terms from both groups.
    - **Exclude:** Filters out any article containing these terms.

    The resulting Boolean query is automatically constructed as:  
    `(Group 1 term OR ...) AND (Group 2 term OR ...) AND NOT (Excluded terms)`

    You can choose to include only specific types of studies (e.g., clinical trials, meta-analyses, longitudinal studies) to focus your search.
    """)

    example_query = st.radio("See example search query:", options=["Yes", "No"])
    
    if example_query == "Yes":
        included_1 = st_tags(
            value=adh_terms,
            label="#### Include words or phrases:",
            text="Press enter to add more",
            key='included1_yes'
        )
        included_2 = st_tags(
            value=breast_cancer_terms,
            label="#### Include words or phrases:",
            text="Press enter to add more",
            key='included2_yes'
        )
    else:
        included_1 = st_tags(
            value=[],
            label="#### Include words or phrases:",
            text="Press enter to add more",
            key='included1_no'
        )
        included_2 = st_tags(
            value=[],
            label="#### Include words or phrases:",
            text="Press enter to add more",
            key='included2_no'
        )

    # Always show this regardless of query example
    excluded = st_tags(
        label="##### Exclude words or phrases:",
        text="Press enter to add more",
        maxtags=4,
        key='excluded'
    )

    query_inc1 = " OR ".join([f'"{term}"' for term in included_1])
    query_inc2 = " OR ".join([f'"{term}"' for term in included_2])
    query_ex = " AND ".join([f'NOT "{term}"' for term in excluded])
    query = f'({query_inc1} AND {query_inc2}) {query_ex}'

    st.write('##### Final Query')
    st.write(query)

    # Query options
    c3, c4, c5, c6, _, c9 = st.columns([3, 4, 4, 4, 8, 3])
    with c3:
        max_results = st.number_input('Maximum Results', min_value=500, max_value=500000, value=10000, step=100)
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

    if run_query:
        if not included_1:
            st.error("Query is empty!")
        else:
            placeholder = st.empty()
            placeholder.write("Searching PubMed...")
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

st.markdown("""
Clustering groups similar articles together based on abstract content, allowing researchers to explore common themes.  
Dimensionality reduction projects the high-dimensional article features (e.g., TF-IDF vectors) into a 2D space for visualization.""")
c20, c21 = st.columns([10,10])
with c20:
    st.markdown("""
    | Clustering Algorithm          | Description |
    |--------------------|-------------|
    | **K-Means**        | Partitions data into a predefined number of clusters by minimizing variance. |
    | **LDA**            | Topic modeling approach that infers latent topics from abstract text. |
    | **DBSCAN**         | Density-based method that finds clusters of high density and identifies noise. |
    | **Hierarchical**   | Builds nested clusters using linkage distances.
    """)
with c21:
    st.markdown("""
    | Dimensionality Reduction Method             | Purpose |
    |--------------------|---------|
    | **PCA**            | Linear projection to maximize variance. |
    | **t-SNE**          | Preserves local structure (good for small to mid-sized datasets). |
    | **UMAP**           | Balances global and local structure with faster computation and better scalability.
    """)

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
    cluster_btn = st.button("Begin Clustering", type='primary')

if cluster_btn:
    st.session_state.pop("cluster_df", None)
    st.session_state.pop("cluster_plot", None)
    st.session_state.pop("clustered_data", None)
    
    if st.session_state.df is None or st.session_state.df.empty:
        st.error("No data to cluster!")
    else:
        cluster_sizes, cluster_keywords, cluster_keyword_scores, centroids = run_cluster(
            st.session_state.df, n_clusters=num_clusters, n_key_words=n_keywords,
            dimred_method=dimred_method, cluster_method=cluster_method
        )
        # Generate the interactive plot using Plotly
        st.write("### Updated Search Results with Clusters", st.session_state.df)
        csv_clustered = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="Download clustered results as CSV",
            data=csv_clustered,
            file_name='pubmed_clustered_results.csv',
            mime='text/csv'
        )
        st.divider()
        # Merge cluster sizes and keywords into one DataFrame
        cluster_df = pd.DataFrame.from_dict(cluster_keywords, orient='index',
                                            columns=[f"Term {i+1}" for i in range(n_keywords)])
        cluster_df.index.name = "Cluster"
        cluster_df = cluster_df.reset_index()

        # Add count column from cluster sizes
        cluster_df["Count"] = cluster_df["Cluster"].map(cluster_sizes)

        # Reorder columns to show Cluster, Count, then Terms
        cols = ["Cluster", "Count"] + [f"Term {i+1}" for i in range(n_keywords)]
        cluster_df = cluster_df[cols]

        # Display the merged table
        st.write("### Cluster Summary")

        st.markdown("""
        ### Interpreting the Cluster Summary Table

        Each cluster is labeled and sorted by the number of articles it contains.  
        For each cluster, the most distinctive keywords are extracted using TF-IDF or topic model weights.

        - These keywords do not overlap across clusters to ensure clarity.
        - Keywords help characterize the central theme of each group.
        """)

        st.dataframe(cluster_df, use_container_width=True)
        st.divider()
        # Display the interactive Plotly chart
        st.session_state.df["Citation"] = (
            st.session_state.df["Authors"].fillna("") + " (" +
            st.session_state.df["PublicationYear"].fillna("N/A") + "). " +
            st.session_state.df["Title"].fillna("")
        )
        fig = plot_dimred_interactive(st.session_state.df, dimred_method, cluster_method, centroids)
        st.plotly_chart(fig, use_container_width=False)
        fig.update_layout(width=1000, height=700)

        st.session_state.cluster_df = cluster_df
        st.session_state.cluster_plot = fig
        st.session_state.clustered_data = st.session_state.df.copy()

st.divider()
st.subheader("Summarization")

st.markdown("""
### Cluster Summarization

Using the Hugging Face `facebook/bart-large-cnn` model, each cluster of abstracts is summarized into a few sentences that describe the dominant theme.

- This step provides a quick synopsis of each topic area.
- Summaries are generated from all abstracts in a cluster, truncated to fit API limits.

⚠️ **Note:** This step uses remote inference and may incur latency for large clusters.
""")

num_sentences = st.slider("Sentences per summary", 1, 5, 3)
sum_bool = st.button("Summarize Clusters", type='primary')

if sum_bool:
    # Re-display the clustered table and plot
    if "cluster_df" in st.session_state:
        st.subheader("Cluster Summary")
        st.dataframe(st.session_state.cluster_df, use_container_width=True)

    if "cluster_plot" in st.session_state:
        st.plotly_chart(st.session_state.cluster_plot, use_container_width=False)

    if "clustered_data" in st.session_state:
        st.write("### Clustered Data Table")
        st.dataframe(st.session_state.clustered_data, use_container_width=True)

    clust_sum = []
    for cluster in sorted(st.session_state.df["Cluster"].unique()):
        cluster_text = " ".join(st.session_state.df[st.session_state.df["Cluster"] == cluster]["Abstract"].dropna().tolist())
        cluster_text = cluster_text[:2000]  # Truncate to avoid hitting Hugging Face API limits
        summary = summarize_huggingface_api(cluster_text, hf_token=st.secrets["hf_token"])
        clust_sum.append((cluster, summary))

    for cluster, summary in clust_sum:
        st.markdown(f"**Cluster {cluster} Summary**")
        st.write(summary)
