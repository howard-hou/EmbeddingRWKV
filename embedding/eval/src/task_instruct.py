import json, time, random, os
import numpy as np
import dataclasses
from torch.nn import functional as F
from typing import List, Dict
from collections import defaultdict
from PIL import Image
from pathlib import Path
import logging


def _format_instruction(description: str) -> str:
    """Return instruction string in the shared `Instruct/Query` template."""
    return f"Instruct: {description}\nQuery: {{query}}"


def get_default_instruct(task_description: str) -> str:
    if not task_description:
        return ''
    return _format_instruction(task_description)


DEFAULT_FALLBACK_BY_TYPE: Dict[str, str] = {
    'Classification': 'Categorizing the given news title',
    'Clustering': 'Categorizing the given news title',
    'Retrieval': 'Given a query, retrieve documents that answer the query',
    'Reranking': 'Given a query, retrieve documents that answer the query',
    'STS': 'Retrieve semantically similar text',
    'PairClassification': 'Retrieve semantically similar text',
}


def get_classification_task_instruct(task_name: str) -> str:
    task_name_to_instruct: Dict[str, str] = {
        'AmazonCounterfactualClassification': 'Given an Amazon review, judge whether it is counterfactual.',
        'AmazonPolarityClassification': 'Classifying Amazon reviews into positive or negative sentiment',
        'AmazonReviewsClassification': 'Classifying the given Amazon review into its appropriate rating category',
        'Banking77Classification': 'Given a online banking query, find the corresponding intents',
        'EmotionClassification': 'Classifying the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise',
        'ImdbClassification': 'Classifying the sentiment expressed in the given movie review text from the IMDB dataset',
        'MassiveIntentClassification': 'Given a user utterance as query, find the user intents',
        'MassiveScenarioClassification': 'Given a user utterance as query, find the user scenarios',
        'MTOPDomainClassification': 'Classifying the intent domain of the given utterance in task-oriented conversation',
        'MTOPIntentClassification': 'Classifying the intent of the given utterance in task-oriented conversation',
        'ToxicConversationsClassification': 'Classifying the given comments as either toxic or not toxic',
        'TweetSentimentExtractionClassification': 'Classifying the sentiment of a given tweet as either positive, negative, or neutral',
        'TNews': 'Categorizing the given news title',
        'IFlyTek': 'Given an App description text, find the appropriate fine-grained category',
        'MultilingualSentiment': 'Classifying sentiment of the customer review into positive, neutral, or negative',
        'JDReview': 'Classifying sentiment of the customer review for iPhone into positive or negative',
        'OnlineShopping': 'Classifying sentiment of the customer review into positive or negative',
        'Waimai': 'Classify the customer review from a food takeaway platform into positive or negative',
    }
    if task_name not in task_name_to_instruct:
        logging.warning(f"No instruction config for task {task_name}, use fallback instruction from task_instruction.tex.")
        return get_default_instruct(DEFAULT_FALLBACK_BY_TYPE['Classification'])
    return get_default_instruct(task_name_to_instruct[task_name])


def get_clustering_task_instruct(task_name: str) -> str:
    task_name_to_instruct: Dict[str, str] = {
        'ArxivClusteringP2P': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
        'ArxivClusteringS2S': 'Identify the main and secondary category of Arxiv papers based on the titles',
        'BiorxivClusteringP2P': 'Identify the main category of Biorxiv papers based on the titles and abstracts',
        'BiorxivClusteringS2S': 'Identify the main category of Biorxiv papers based on the titles',
        'MedrxivClusteringP2P': 'Identify the main category of Medrxiv papers based on the titles and abstracts',
        'MedrxivClusteringS2S': 'Identify the main category of Medrxiv papers based on the titles',
        'RedditClustering': 'Identify the topic or theme of Reddit posts based on the titles',
        'RedditClusteringP2P': 'Identify the topic or theme of Reddit posts based on the titles and posts',
        'StackExchangeClustering': 'Identify the topic or theme of StackExchange posts based on the titles',
        'StackExchangeClusteringP2P': 'Identify the topic or theme of StackExchange posts based on the given paragraphs',
        'TwentyNewsgroupsClustering': 'Identify the topic or theme of the given news articles',
        'CLSClusteringS2S': 'Identify the main category of scholar papers based on the titles',
        'CLSClusteringP2P': 'Identify the main category of scholar papers based on the titles and abstracts',
        'ThuNewsClusteringS2S': 'Identify the topic or theme of the given news articles based on the titles',
        'ThuNewsClusteringP2P': 'Identify the topic or theme of the given news articles based on the titles and contents',
    }
    if task_name not in task_name_to_instruct:
        logging.warning(f"No instruction config for task {task_name}, use fallback instruction from task_instruction.tex.")
        return get_default_instruct(DEFAULT_FALLBACK_BY_TYPE['Clustering'])
    return get_default_instruct(task_name_to_instruct[task_name])

def get_task_instruct_by_task(task) -> str:
    task_name = Path(task.metadata.name).stem # remove version suffix if any
    task_type = task.metadata.type
    # task-specific overrides based on detailed instruction list
    pair_classification_overrides = {
        'SprintDuplicateQuestions': 'Retrieve semantically similar questions',
    }
    reranking_overrides = {
        'AskUbuntuDupQuestions': 'Retrieve semantically similar questions',
        'StackOverflowDupQuestions': 'Retrieve semantically similar questions',
        'SciDocsRR': 'Retrieve relevant paper titles',
    }
    retrieval_overrides = {
        'QuoraRetrieval': 'Retrieve semantically similar questions',
        'CQADupstack': 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question',
    }
    summary_overrides = {
        'SummEval': 'Retrieve semantically similar summaries',
    }
    if task_type == 'Classification':
        return get_classification_task_instruct(task_name)
    if task_type == 'Clustering':
        return get_clustering_task_instruct(task_name)
    if task_type == 'Retrieval':
        description = retrieval_overrides.get(task_name, DEFAULT_FALLBACK_BY_TYPE['Retrieval'])
        return get_default_instruct(description)
    if task_type == 'Reranking':
        description = reranking_overrides.get(task_name, DEFAULT_FALLBACK_BY_TYPE['Reranking'])
        return get_default_instruct(description)
    if task_type == 'STS':
        return get_default_instruct(DEFAULT_FALLBACK_BY_TYPE['STS'])
    if task_type == 'BitextMining':
        return get_default_instruct('Given two sentences, judge whether they are translations of each other')
    if task_type == 'PairClassification':
        description = pair_classification_overrides.get(task_name, DEFAULT_FALLBACK_BY_TYPE['PairClassification'])
        return get_default_instruct(description)
    if task_type == 'MultilabelClassification':
        return get_default_instruct('Given a text, classify it into one or more categories')
    if task_type == 'Regression':
        return get_default_instruct('Given a text, predict its score')
    if task_type == 'InstructionRetrieval':
        return get_default_instruct('Given a query, retrieve documents that answer the query')
    if task_type == 'Summarization':
        description = summary_overrides.get(task_name, 'Retrieve semantically similar summaries')
        return get_default_instruct(description)
    if task_type == 'Speed':
        return get_default_instruct('Given a text, encode it into a fixed-size vector representation')
    logging.warning(f"Unknown task type {task_type}, use none instruction.")
    return get_default_instruct(None)
