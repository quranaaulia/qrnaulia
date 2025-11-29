import os
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import plotly.express as px
import joblib
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

def load_bertopic_model(model_path='models/bertopic_model.pkl'):
    """Load topic model"""
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except:
        pass
    return None

def build_bertopic_model(filepath=None):
    """Build topic model using BERTopic from uploaded CSV data"""
    if filepath is None:
        return {
            'error': 'Filepath tidak diberikan. Pastikan file CSV sudah diupload.'
        }

    try:
        # Load and preprocess data
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        from services.preprocessing import get_preprocessing_steps
        preprocessing_results = get_preprocessing_steps(df)
        text_data = [item['text_clean'] for item in preprocessing_results['hasil_preprocessing'] if item['text_clean']]

        if not text_data:
            return {
                'error': 'Tidak ada data teks yang valid setelah preprocessing.'
            }

        # Limit data for performance (take first 500 samples to prevent crash)
        max_samples = 500
        if len(text_data) > max_samples:
            text_data = text_data[:max_samples]

        # Setup Indonesian stopwords
        indonesian_stopwords = stopwords.words('indonesian')

        # Create BERTopic model with optimized settings for performance
        # Use lightweight embedding model
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # Optimized UMAP for lower memory usage
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        # HDBSCAN with optimized parameters
        hdbscan_model = HDBSCAN(
            min_cluster_size=10,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # Vectorizer with Indonesian stopwords
        vectorizer_model = CountVectorizer(
            stop_words=indonesian_stopwords,
            ngram_range=(1, 2),
            min_df=3
        )

        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=KeyBERTInspired(),
            ctfidf_model=ClassTfidfTransformer(),
            nr_topics="auto",  # Auto-determine number of topics
            verbose=True
        )

        # Fit the model
        topics, probs = topic_model.fit_transform(text_data)

        # Get topic information
        topic_info = topic_model.get_topic_info()

        # Filter out outlier topic (-1)
        topic_info = topic_info[topic_info['Topic'] != -1]

        # Create topics summary
        topics_summary = []
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            count = row['Count']
            name = row['Name']

            # Get top words for this topic
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                keywords = [word for word, _ in topic_words[:5]]  # Top 5 words
            else:
                keywords = []

            # Calculate probability safely
            topic_probs = probs[topics == topic_id]
            probability = topic_probs.mean() if len(topic_probs) > 0 else 0.0

            topics_summary.append({
                'topic_id': topic_id,
                'keywords': keywords,
                'count': count,
                'probability': probability,
                'name': name
            })

        # Calculate coherence score using BERTopic's built-in method (if available)
        try:
            coherence_score = topic_model.get_coherence_score()
        except AttributeError:
            # Fallback for older BERTopic versions that don't have get_coherence_score
            coherence_score = 0.0

        # Save model
        model_data = {
            'topic_model': topic_model,
            'topics_summary': topics_summary,
            'coherence_score': coherence_score,
            'total_topics': len(topics_summary),
            'topics': topics,
            'probs': probs
        }
        joblib.dump(model_data, 'models/bertopic_model.pkl')

        # Create visualizations using BERTopic's built-in methods
        try:
            # Barchart visualization (word frequencies per topic) - this is the main BERTopic barchart
            fig_barchart = topic_model.visualize_barchart(top_n_topics=10, height=400)
            barchart_html = fig_barchart.to_html(full_html=False) if fig_barchart else "<p>Visualisasi barchart tidak tersedia</p>"

            # Create simple distribution chart using plotly (topic counts)
            topic_ids = [f'Topik {topic["topic_id"]}' for topic in topics_summary[:10]]  # Top 10 topics
            counts = [topic['count'] for topic in topics_summary[:10]]

            fig_dist = px.bar(
                x=topic_ids,
                y=counts,
                title='Distribusi Topik (BERTopic)',
                labels={'x': 'Topik', 'y': 'Jumlah Komentar'},
                color=counts,
                color_continuous_scale='Blues'
            )
            fig_dist.update_layout(height=400)
            distribution_html = fig_dist.to_html(full_html=False)

            # Topics visualization (2D scatter plot)
            fig_topics = topic_model.visualize_topics(height=400)
            topics_html = fig_topics.to_html(full_html=False) if fig_topics else "<p>Visualisasi topik 2D tidak tersedia</p>"

            # Hierarchy visualization
            if len(topic_info_filtered) > 5:
                fig_hierarchy = topic_model.visualize_hierarchy(height=400)
                hierarchy_html = fig_hierarchy.to_html(full_html=False) if fig_hierarchy else "<p>Visualisasi hierarchy tidak tersedia</p>"
            else:
                hierarchy_html = "<p>Visualisasi hierarchy memerlukan minimal 5 topik</p>"

        except Exception as viz_error:
            # Fallback visualizations if BERTopic viz fails
            barchart_html = "<p>Visualisasi barchart gagal dimuat</p>"
            distribution_html = "<p>Visualisasi distribusi gagal dimuat</p>"
            topics_html = "<p>Visualisasi topik 2D gagal dimuat</p>"
            hierarchy_html = "<p>Visualisasi hierarchy gagal dimuat</p>"

        return {
            'topics_summary': topics_summary,
            'coherence_score': coherence_score,
            'total_topics': len(topics_summary),
            'model_type': 'BERTopic (Real Implementation)',
            'visualizations': {
                'barchart': distribution_html,
                'hierarchy': hierarchy_html,
                'topics': topics_html,
                'distribution': distribution_html  # Same as barchart for compatibility
            }
        }

    except Exception as e:
        return {
            'error': f'Gagal membangun model BERTopic: {str(e)}'
        }

def get_bertopic_analysis():
    """Get topic analysis data from saved BERTopic model"""
    model_data = load_bertopic_model()
    if model_data is None:
        return {
            'error': 'Model topik tidak ditemukan. Pastikan model sudah dibangun.'
        }

    topic_model = model_data['topic_model']
    topics_summary = model_data['topics_summary']

    # Create visualizations using BERTopic's built-in methods
    try:
        # Barchart visualization (word frequencies per topic)
        fig_barchart = topic_model.visualize_barchart(top_n_topics=10, height=400)
        barchart_html = fig_barchart.to_html(full_html=False) if fig_barchart else "<p>Visualisasi barchart tidak tersedia</p>"

        # Create simple distribution chart using plotly (topic counts)
        topic_ids = [f'Topik {topic["topic_id"]}' for topic in topics_summary[:10]]  # Top 10 topics
        counts = [topic['count'] for topic in topics_summary[:10]]

        fig_dist = px.bar(
            x=topic_ids,
            y=counts,
            title='Distribusi Topik (BERTopic)',
            labels={'x': 'Topik', 'y': 'Jumlah Komentar'},
            color=counts,
            color_continuous_scale='Blues'
        )
        fig_dist.update_layout(height=400)
        distribution_html = fig_dist.to_html(full_html=False)

        # Topics visualization (2D scatter plot)
        fig_topics = topic_model.visualize_topics(height=400)
        topics_html = fig_topics.to_html(full_html=False) if fig_topics else "<p>Visualisasi topik 2D tidak tersedia</p>"

        # Hierarchy visualization
        topic_info = topic_model.get_topic_info()
        topic_info_filtered = topic_info[topic_info['Topic'] != -1]
        if len(topic_info_filtered) > 5:
            fig_hierarchy = topic_model.visualize_hierarchy(height=400)
            hierarchy_html = fig_hierarchy.to_html(full_html=False) if fig_hierarchy else "<p>Visualisasi hierarchy tidak tersedia</p>"
        else:
            hierarchy_html = "<p>Visualisasi hierarchy memerlukan minimal 5 topik</p>"

    except Exception as viz_error:
        # Fallback visualizations if BERTopic viz fails
        barchart_html = "<p>Visualisasi barchart gagal dimuat</p>"
        distribution_html = "<p>Visualisasi distribusi gagal dimuat</p>"
        topics_html = "<p>Visualisasi topik 2D gagal dimuat</p>"
        hierarchy_html = "<p>Visualisasi hierarchy gagal dimuat</p>"

    return {
        'topics_summary': topics_summary,
        'coherence_score': model_data.get('coherence_score', 0.0),
        'total_topics': model_data.get('total_topics', 0),
        'model_type': 'BERTopic (Real Implementation)',
        'visualizations': {
            'barchart': barchart_html,
            'hierarchy': hierarchy_html,
            'topics': topics_html,
            'distribution': distribution_html
        }
    }
