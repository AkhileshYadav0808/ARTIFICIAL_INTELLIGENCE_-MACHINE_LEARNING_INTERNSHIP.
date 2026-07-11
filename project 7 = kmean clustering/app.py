import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

st.set_page_config(page_title="KMeans Clustering App", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4B0082;'>Customer Segmentation with KMeans</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #36454F;'>Uncovering customer groups based on Age and Income using unsupervised learning.</p>", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data # Cache data loading for performance
def load_data():
    try:
        df = pd.read_csv("income.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'income.csv' not found. Please upload the file.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    st.subheader("Original Data Sample")
    st.dataframe(df.head())

    # --- Preprocessing ---
    st.sidebar.header("Clustering Parameters")
    st.sidebar.markdown("Adjust the number of clusters and view results.")

    st.subheader("Data Preprocessing: Scaling Features")
    st.write("Age and Income features are scaled using MinMaxScaler to ensure fair distance calculation in KMeans.")

    scaler = MinMaxScaler()
    df['Income($)'] = scaler.fit_transform(df[['Income($)']])
    df['Age'] = scaler.fit_transform(df[['Age']])

    st.subheader("Scaled Data Sample")
    st.dataframe(df.head())

    # --- Elbow Method for Optimal K ---
    st.header("1. Elbow Plot: Finding the Optimal Number of Clusters (K)")
    st.write("The elbow method helps determine the best value for 'K' by looking for a point where the decrease in Sum of Squared Errors (SSE) begins to slow down significantly.")

    sse = []
    k_rng = range(1, 10)
    for k in k_rng:
        km = KMeans(n_clusters=k, n_init='auto', random_state=0)
        km.fit(df[['Age', 'Income($)']])
        sse.append(km.inertia_)

    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 5))
    ax_elbow.plot(k_rng, sse, marker='o')
    ax_elbow.set_xlabel('Number of Clusters (K)')
    ax_elbow.set_ylabel('Sum of Squared Error (SSE)')
    ax_elbow.set_title('Elbow Method for Optimal K')
    ax_elbow.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_elbow)

    st.markdown("--- ")

    # --- KMeans Clustering ---
    st.header("2. KMeans Clustering Results")

    # Interactive K selection
    k_selected = st.sidebar.slider("Select K (Number of Clusters)", min_value=2, max_value=5, value=3)

    km = KMeans(n_clusters=k_selected, n_init='auto', random_state=0)
    y_predicted = km.fit_predict(df[['Age', 'Income($)']])
    df['cluster'] = y_predicted

    st.subheader(f"Clusters Formed (K={k_selected})")
    st.dataframe(df.head())

    # --- Visualization of Clusters ---
    st.subheader("Visualizing Customer Clusters")
    st.write("Each color represents a different customer segment, with purple stars indicating cluster centroids.")

    fig_clusters, ax_clusters = plt.subplots(figsize=(10, 6))

    colors = ['green', 'red', 'black', 'blue', 'orange']
    for i in range(k_selected):
        cluster_df = df[df.cluster == i]
        ax_clusters.scatter(cluster_df['Age'], cluster_df['Income($)'], color=colors[i % len(colors)], label=f'Cluster {i}')

    # Plotting centroids
    ax_clusters.scatter(km.cluster_centers_[:, 0],
                        km.cluster_centers_[:, 1],
                        color='purple', marker='*', s=200, label='Centroids', edgecolor='white')

    ax_clusters.set_xlabel('Age (Scaled)')
    ax_clusters.set_ylabel('Income ($) (Scaled)')
    ax_clusters.set_title(f'Customer Clusters (K={k_selected})')
    ax_clusters.legend()
    ax_clusters.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_clusters)

    st.subheader("Cluster Centroids")
    st.write("These are the average Age and Income for each identified cluster.")
    centroids_df = pd.DataFrame(km.cluster_centers_, columns=['Centroid Age (Scaled)', 'Centroid Income ($) (Scaled)'])
    centroids_df.index.name = 'Cluster'
    st.dataframe(centroids_df)

    st.markdown("--- ")
    st.markdown("### Deployment Instructions")
    st.info(
        """To run this Streamlit application:\n
        1.  Save the code above into a file named `app.py` in your local machine.\n
        2.  Open your terminal or command prompt.\n
        3.  Navigate to the directory where you saved `app.py`.\n
        4.  Run `streamlit run app.py`.\n
        This will open the application in your web browser. You can then use Streamlit Community Cloud or other platforms ."""
    )
else:
    st.warning("Please ensure 'income.csv' is available to run the clustering analysis.")
