import streamlit as st
import pandas as pd
import os
from config import Config
from utils.file_utils import allowed_file
import tempfile
from services.preprocessing import get_preprocessing_steps
from services.word2vec_service import get_word2vec_analysis, load_word2vec_model, get_similar_words
from services.bertopic_service import build_bertopic_model, get_bertopic_analysis

# Set page config
st.set_page_config(page_title="TikTok Comments Analyzer", page_icon="📊", layout="wide")

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Initialize session state for BERTopic
if 'bertopic_built' not in st.session_state:
    st.session_state.bertopic_built = False

if 'bertopic_data' not in st.session_state:
    st.session_state.bertopic_data = None

def main():
    st.title("📊 TikTok Comments Analyzer")

    # Check if file is uploaded
    if st.session_state.uploaded_file and os.path.exists(st.session_state.uploaded_file):
        # Sidebar navigation (without Upload option)
        page = st.sidebar.selectbox(
            "Pilih Menu",
            ["Home", "Komentar Asli", "Komentar Preprocessing", "Word2Vec", "BERTopic", "Analysis"]
        )

        if page == "Home":
            home_page()
        elif page == "Komentar Asli":
            comments_raw_page()
        elif page == "Komentar Preprocessing":
            comments_preprocessed_page()
        elif page == "Word2Vec":
            word2vec_page()
        elif page == "BERTopic":
            bertopic_page()
        elif page == "Analysis":
            analysis_page()
    else:
        # No file uploaded, show only upload page without sidebar
        upload_page()

def upload_page():
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Pilih file CSV komentar TikTok", type=['csv'])

    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.uploaded_file = tmp_file.name

        # Reset BERTopic session state for new file
        st.session_state.bertopic_built = False
        st.session_state.bertopic_data = None

        st.success("File berhasil diupload!")
        st.info("Sedang memuat halaman Home...")
        st.rerun()

def home_page():
    st.header("Dashboard")

    if st.session_state.uploaded_file and os.path.exists(st.session_state.uploaded_file):
        st.success("File sudah diupload dan siap untuk analisis.")

        # Option to upload new file
        st.subheader("Upload File Baru (Opsional)")
        new_file = st.file_uploader("Pilih file CSV baru", type=['csv'], key="new_upload")
        if new_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(new_file.getvalue())
                st.session_state.uploaded_file = tmp_file.name
            st.success("File berhasil diganti!")
    else:
        st.warning("Silakan upload file CSV terlebih dahulu melalui menu Upload.")

def comments_raw_page():
    st.header("💬 Komentar Asli")

    if not st.session_state.uploaded_file or not os.path.exists(st.session_state.uploaded_file):
        st.warning("Silakan upload file CSV terlebih dahulu.")
        return

    try:
        df = pd.read_csv(st.session_state.uploaded_file, encoding='utf-8-sig')
    except Exception as e:
        st.error(f'Gagal membaca file CSV: {e}')
        return

    # Normalisasi nama kolom
    columns_map = {c.strip().lower(): c for c in df.columns}
    if 'text' not in columns_map or 'createtimeiso' not in columns_map:
        st.error(f'File CSV harus memiliki kolom "text" dan "createTimeISO". Kolom ditemukan: {df.columns.tolist()}')
        return

    text_col = columns_map['text']
    date_col = columns_map['createtimeiso']

    comments = [
        {'text': str(row[text_col]), 'date': str(row[date_col])}
        for _, row in df[[text_col, date_col]].dropna(subset=[text_col]).iterrows()
    ]

    # Statistics Section
    st.subheader("📊 Statistik Komentar")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Komentar", len(comments))

    with col2:
        avg_length = sum(len(c['text']) for c in comments) / len(comments) if comments else 0
        st.metric("Rata-rata Panjang", f"{avg_length:.1f} karakter")

    with col3:
        max_length = max(len(c['text']) for c in comments) if comments else 0
        st.metric("Komentar Terpanjang", f"{max_length} karakter")

    with col4:
        min_length = min(len(c['text']) for c in comments) if comments else 0
        st.metric("Komentar Terpendek", f"{min_length} karakter")

    # Search and Filter Section
    st.subheader("🔍 Cari & Filter")
    col_search, col_limit = st.columns([2, 1])

    with col_search:
        search_term = st.text_input("Cari komentar:", placeholder="Ketik kata kunci...")

    with col_limit:
        display_limit = st.selectbox(
            "Jumlah komentar:",
            [10, 25, 50, 100, len(comments)],
            index=2,
            format_func=lambda x: f"{x} komentar" if x != len(comments) else "Semua komentar"
        )

    # Filter comments based on search
    if search_term:
        filtered_comments = [
            comment for comment in comments
            if search_term.lower() in comment['text'].lower()
        ]
        st.info(f"Ditemukan {len(filtered_comments)} komentar yang mengandung '{search_term}'")
    else:
        filtered_comments = comments

    # Display comments
    st.subheader("📝 Daftar Komentar")

    if not filtered_comments:
        st.warning("Tidak ada komentar yang sesuai dengan kriteria pencarian.")
        return

    # Show only selected limit
    comments_to_show = filtered_comments[:display_limit]

    for i, comment in enumerate(comments_to_show, 1):
        # Create a container for each comment
        with st.container():
            # Date header
            st.markdown(f"**📅 {comment['date'][:10]}** - *{comment['date']}*")

            # Comment text in a styled box
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0;">
                {comment['text']}
            </div>
            """, unsafe_allow_html=True)

            # Comment length info
            st.caption(f"📏 Panjang: {len(comment['text'])} karakter")

            # Separator
            st.markdown("---")

    # Pagination info
    if len(filtered_comments) > display_limit:
        st.info(f"Menampilkan {display_limit} dari {len(filtered_comments)} komentar yang sesuai.")
    else:
        st.success(f"Menampilkan semua {len(filtered_comments)} komentar yang sesuai.")

def comments_preprocessed_page():
    st.header("🔧 Komentar Setelah Preprocessing")

    if not st.session_state.uploaded_file or not os.path.exists(st.session_state.uploaded_file):
        st.warning("Silakan upload file CSV terlebih dahulu.")
        return

    try:
        df = pd.read_csv(st.session_state.uploaded_file, encoding='utf-8-sig')

        # Get preprocessing steps
        preprocessing_results = get_preprocessing_steps(df)

        original_comments = df['text'].dropna().tolist()
        preprocessed_comments = preprocessing_results['hasil_preprocessing']

        # Statistics Comparison Section
        st.subheader("📊 Perbandingan Statistik")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Komentar Asli", len(original_comments))
            st.metric("Total Setelah Preprocessing", len(preprocessed_comments), delta=len(preprocessed_comments)-len(original_comments))

        with col2:
            avg_orig = sum(len(c) for c in original_comments) / len(original_comments) if original_comments else 0
            avg_prep = sum(len(c) for c in preprocessed_comments) / len(preprocessed_comments) if preprocessed_comments else 0
            st.metric("Rata-rata Panjang Asli", f"{avg_orig:.1f} karakter")
            st.metric("Rata-rata Panjang Preprocessing", f"{avg_prep:.1f} karakter", delta=f"{avg_prep-avg_orig:.1f}")

        with col3:
            max_orig = max(len(c) for c in original_comments) if original_comments else 0
            max_prep = max(len(c) for c in preprocessed_comments) if preprocessed_comments else 0
            st.metric("Komentar Terpanjang Asli", f"{max_orig} karakter")
            st.metric("Komentar Terpanjang Preprocessing", f"{max_prep} karakter", delta=max_prep-max_orig)

        with col4:
            min_orig = min(len(c) for c in original_comments) if original_comments else 0
            min_prep = min(len(c) for c in preprocessed_comments) if preprocessed_comments else 0
            st.metric("Komentar Terpendek Asli", f"{min_orig} karakter")
            st.metric("Komentar Terpendek Preprocessing", f"{min_prep} karakter", delta=min_prep-min_orig)

        # Comparison Table Section
        st.subheader("📋 Perbandingan Komentar")

        # Create comparison table
        comparison_data = []
        for original, processed in zip(original_comments, preprocessed_comments):
            # Clean the processed comment to remove unwanted text and symbols
            cleaned_processed = str(processed).strip('{}').replace('"text":', '').strip('"').strip()
            comparison_data.append({
                "Komentar Asli": original,
                "Komentar Setelah Preprocessing": cleaned_processed
            })

        # Display as dataframe with custom styling
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(
            df_comparison,
            column_config={
                "Komentar Asli": st.column_config.TextColumn("Komentar Asli", width="large"),
                "Komentar Setelah Preprocessing": st.column_config.TextColumn("Komentar Setelah Preprocessing", width="large")
            },
            hide_index=True,
            use_container_width=True
        )

        # Add custom CSS for larger font sizes
        st.markdown("""
        <style>
        .dataframe {
            font-size: 18px !important;
        }
        .dataframe th {
            font-size: 20px !important;
            font-weight: bold !important;
        }
        .dataframe td {
            font-size: 18px !important;
        }
        .metric-container {
            font-size: 20px !important;
        }
        .metric-value {
            font-size: 24px !important;
            font-weight: bold !important;
        }
        .metric-delta {
            font-size: 18px !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-size: 28px !important;
        }
        .stMarkdown p {
            font-size: 18px !important;
        }
        .stTextInput input {
            font-size: 16px !important;
        }
        .stSelectbox select {
            font-size: 16px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f'Error dalam preprocessing: {str(e)}')

def word2vec_page():
    st.header("🔤 Word2Vec Analysis")

    if not st.session_state.uploaded_file or not os.path.exists(st.session_state.uploaded_file):
        st.warning("Silakan upload file CSV terlebih dahulu.")
        return

    try:
        data = get_word2vec_analysis(st.session_state.uploaded_file)

        if 'error' in data:
            st.error(data['error'])
            return

        # Model Information Section
        st.subheader("📊 Informasi Model")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Ukuran Vokabuler", f"{data['model_info']['vocab_size']}")
        with col2:
            st.metric("Dimensi Vektor", f"{data['model_info']['vector_size']}")
        with col3:
            st.metric("Ukuran Window", f"{data['model_info']['window_size']}")
        with col4:
            st.metric("Min Count", f"{data['model_info']['min_count']}")
        with col5:
            st.metric("Workers", f"{data['model_info']['workers']}")

        # Most Frequent Words Section
        st.subheader("🔍 Kata yang Sering Muncul")
        if data['sample_embeddings']:
            # Create table data for most frequent words
            frequent_words_data = []
            for i, embedding in enumerate(data['sample_embeddings'], 1):
                # Format embedding vector (first 2 dimensions to match 2D visualization)
                embedding_str = f"[{embedding['vector'][0]:.3f}, {embedding['vector'][1]:.3f}]"
                frequent_words_data.append({
                    "Kata": embedding['word'],
                    "Embedding (2D)": embedding_str,
                    "Frekuensi": f"#{i}"  # Since index_to_key is ordered by frequency
                })

            # Display as dataframe
            df_frequent = pd.DataFrame(frequent_words_data)
            st.dataframe(
                df_frequent,
                column_config={
                    "Kata": st.column_config.TextColumn("Kata", width="medium"),
                    "Embedding (2D)": st.column_config.TextColumn("Embedding (2D)", width="large"),
                    "Frekuensi": st.column_config.TextColumn("Peringkat Frekuensi", width="medium")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Tidak ada data kata yang tersedia.")



        # Visualization Section
        st.subheader("📈 Visualisasi Embedding")
        if data['visualization']:
            st.components.v1.html(data['visualization'], height=650)
        else:
            st.info("Visualisasi tidak tersedia. Pastikan model memiliki cukup kata untuk visualisasi.")

        # Interactive Word Similarity Search
        st.subheader("🔍 Cari Kata Serupa")
        col_search, col_topn = st.columns([2, 1])

        with col_search:
            search_word = st.text_input("Masukkan kata:", placeholder="Ketik kata yang ingin dicari...")

        with col_topn:
            topn = st.selectbox("Jumlah hasil:", [5, 10, 15, 20], index=0)

        if search_word:
            try:
                model = load_word2vec_model()
                if model and search_word in model.wv:
                    similar_words = get_similar_words(model, search_word, topn=topn)
                    if similar_words:
                        st.success(f"Ditemukan {len(similar_words)} kata serupa untuk '{search_word}':")
                        for i, similar in enumerate(similar_words, 1):
                            st.write(f"{i}. **{similar['word']}** - Similarity: {similar['similarity']:.3f}")
                    else:
                        st.warning(f"Tidak ditemukan kata serupa untuk '{search_word}'.")
                else:
                    st.error(f"Kata '{search_word}' tidak ditemukan dalam vokabuler model.")
            except Exception as e:
                st.error(f"Error dalam pencarian: {str(e)}")

    except Exception as e:
        st.error(f'Error dalam analisis Word2Vec: {str(e)}')

def bertopic_page():
    st.header("BERTopic Analysis")

    if not st.session_state.uploaded_file or not os.path.exists(st.session_state.uploaded_file):
        st.warning("Silakan upload file CSV terlebih dahulu.")
        return

    try:
        # Check if model is already built in session
        if st.session_state.bertopic_built and st.session_state.bertopic_data is not None:
            data = st.session_state.bertopic_data
            st.info("Menggunakan model BERTopic yang sudah dibangun sebelumnya.")
        elif os.path.exists('models/bertopic_model.pkl'):
            data = get_bertopic_analysis()
            st.session_state.bertopic_data = data
            st.session_state.bertopic_built = True
            st.info("Model BERTopic dimuat dari file.")
        else:
            with st.spinner("Membangun model BERTopic... Ini mungkin memakan waktu beberapa menit."):
                data = build_bertopic_model(st.session_state.uploaded_file)
            st.session_state.bertopic_data = data
            st.session_state.bertopic_built = True
            st.success("Model BERTopic berhasil dibangun dan disimpan!")

        # Display results
        if 'error' in data:
            st.error(data['error'])
        else:
            st.subheader("📊 Hasil Analisis BERTopic")
            st.write(f"**Model Type:** {data.get('model_type', 'N/A')}")
            st.write(f"**Total Topics:** {data.get('total_topics', 0)}")
            st.write(f"**Coherence Score:** {data.get('coherence_score', 0):.3f}")

            # Display topics summary (Top 10 topics)
            if 'topics_summary' in data:
                st.subheader("🔍 Top 10 Topik Teratas")
                # Sort by count and take top 10
                top_topics = sorted(data['topics_summary'], key=lambda x: x['count'], reverse=True)[:10]
                topics_df = pd.DataFrame(top_topics)
                st.dataframe(
                    topics_df,
                    column_config={
                        "topic_id": st.column_config.NumberColumn("ID Topik"),
                        "keywords": st.column_config.ListColumn("Kata Kunci"),
                        "count": st.column_config.NumberColumn("Jumlah Komentar"),
                        "probability": st.column_config.NumberColumn("Probabilitas", format="%.3f"),
                        "name": st.column_config.TextColumn("Nama Topik")
                    },
                    hide_index=True,
                    use_container_width=True
                )

            # Display visualizations
            if 'visualizations' in data:
                st.subheader("📈 Visualisasi")

                # Barchart - Top 10 topics
                if data['visualizations'].get('barchart'):
                    st.markdown("**Barchart - Kata-Kata Utama per Topik (Top 10):**")
                    st.components.v1.html(data['visualizations']['barchart'], height=400)

                # Topics 2D visualization
                if data['visualizations'].get('topics'):
                    st.markdown("**Visualisasi Topik 2D:**")
                    st.components.v1.html(data['visualizations']['topics'], height=400)

                # Hierarchy visualization
                if data['visualizations'].get('hierarchy'):
                    st.markdown("**Visualisasi Hierarki Topik:**")
                    st.components.v1.html(data['visualizations']['hierarchy'], height=400)

                # Distribution chart
                if data['visualizations'].get('distribution'):
                    st.markdown("**Distribusi Topik:**")
                    st.components.v1.html(data['visualizations']['distribution'], height=400)

    except Exception as e:
        st.error(f'Error dalam analisis BERTopic: {str(e)}')

def analysis_page():
    st.header("Analysis")
    st.write("Halaman analisis tambahan akan ditambahkan di sini.")

if __name__ == "__main__":
    main()
