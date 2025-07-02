**medscrape** is a lightweight Streamlit-based tool for clustering, summarizing, and visualizing PubMed journal abstracts. Researchers can upload their own datasets or scrape PubMed directly to explore emerging themes and trends in biomedical literature using unsupervised machine learning and natural language processing.

---

## 🚀 Features

- 🔎 **Scrape or Upload Data**  
  - Scrape abstracts from PubMed via search query (Entrez API)  
  - Upload `.csv` files with PubMed metadata

- 🧠 **Clustering & Topic Modeling**  
  - Choose from multiple clustering algorithms:  
    - K-Means  
    - DBSCAN  
    - Hierarchical (Agglomerative)  
    - LDA (Latent Dirichlet Allocation)

- 🔻 **Dimensionality Reduction**  
  - Select projection methods:  
    - PCA  
    - t-SNE  
    - UMAP  

- 📊 **Interactive Visualization**  
  - 2D scatter plot of clustered articles using Plotly  
  - Hover to explore metadata for individual articles

- 📝 **Extractive Summarization**  
  - Each cluster is summarized using a local `sklearn` + `TF-IDF` + cosine similarity method  
  - Fast and serverless—no Hugging Face dependency required

---

## 🖼️ Demo

![demo](https://user-images.githubusercontent.com/.../example.png) <!-- optional: insert GIF or screenshot of interface -->

---

## 📂 Required File Format for Upload

If uploading a CSV, your file must include the following columns:

| Column Name     | Description                                 |
|------------------|---------------------------------------------|
| `Title`         | Article title                                |
| `Abstract`      | Abstract text (used for clustering)          |
| `Authors`       | List of authors                              |
| `Journal`       | Journal name                                 |
| `Year`          | Publication year                             |

**Note:** Additional columns are accepted but ignored by default.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/lonespear/medscrape.git
cd medscrape

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
