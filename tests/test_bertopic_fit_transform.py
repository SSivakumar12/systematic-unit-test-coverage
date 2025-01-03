import pytest
import collections
import pandas as pd
from unittest.mock import patch
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic._bertopic import TopicMapper


@pytest.fixture
def cluster_embeddings(reduced_embeddings, documents):
    clustering_model = HDBSCAN(
        min_cluster_size=3, metric="euclidean", cluster_selection_method="eom", prediction_data=True
    )
    clustering_model.fit(reduced_embeddings)
    updated_topic_map = pd.DataFrame(
        {"Document": documents, "ID": range(len(documents)), "Topic": clustering_model.labels_}
    )
    return updated_topic_map, clustering_model.probabilities_


@patch("bertopic._bertopic.BERTopic._extract_embeddings")
@patch("bertopic._bertopic.BERTopic._reduce_dimensionality")
@patch("bertopic._bertopic.BERTopic._cluster_embeddings")
@patch("bertopic._bertopic.BERTopic._extract_topics")
@patch("bertopic._bertopic.BERTopic._save_representative_docs")
def test_fit_transform(
    mock_save_representative_docs,
    mock_extract_topics,
    mock_cluster_embeddings,
    mock_reduce_dimensionality,
    mock_extract_embeddings,
    embedding_model,
    documents,
    document_embeddings,
    reduced_embeddings,
    cluster_embeddings,
):
    #####################################
    # SET-UP of depandcies and patching  #
    #####################################
    mock_extract_embeddings.return_value = document_embeddings
    mock_reduce_dimensionality.return_value = reduced_embeddings
    mock_cluster_embeddings.return_value = cluster_embeddings

    document_mappings = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": None})

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine", random_state=0),
        hdbscan_model=HDBSCAN(
            min_cluster_size=3, metric="euclidean", cluster_selection_method="eom", prediction_data=True, random_state=0
        ),
        calculate_probabilities=False,
    )

    # set-up  private atttributes
    model.topics_ = cluster_embeddings[0]["Topic"].astype(int).to_list()
    model.topic_sizes_ = collections.Counter(cluster_embeddings[0]["Topic"])
    model.topic_mapper_ = TopicMapper(model.topics_)
    topics, prob = model.fit_transform(documents)

    #########################################################################
    #                               TESTING                                 #
    # Assert that the mocked functions were called with the correct arguments
    #########################################################################

    # embeddings
    mock_extract_embeddings.assert_called_once()
    mock_extract_embeddings.call_args[0] == documents
    mock_extract_embeddings.call_args[1] == [None, "document", False]  # kwargs

    # dimensionality-reduction
    mock_reduce_dimensionality.assert_called_once()
    mock_reduce_dimensionality.call_args[0] == [document_embeddings, None]

    # clustering
    mock_cluster_embeddings.assert_called_once()
    mock_cluster_embeddings.call_args[0] == [reduced_embeddings, document_mappings, True]

    # c-tf-idf
    mock_extract_topics.assert_called_once()
    mock_extract_topics.call_args[0] == [document_mappings]
    mock_extract_topics.call_args[1] == [document_embeddings, False, False]

    mock_save_representative_docs.assert_called_once()
    mock_save_representative_docs.call_args[0] == [document_mappings]
