
# Training Data for Instruction Fine-Tuning

> [!NOTE]
> This repository contains raw datasets, all of which have also been formatted for easy training in the [Embedding Model Datasets](https://huggingface.co/collections/sentence-transformers/embedding-model-datasets-6644d7a3673a511914aa7552) collection. We recommend looking there first.

This repository contains training files to train text embedding models, e.g. using [sentence-transformers](https://www.SBERT.net).

## Data Format

All files are in a `jsonl.gz` format: Each line contains a JSON-object that represent one training example.

The JSON objects can come in different formats:
- **Pairs:** `["text1", "text2"]` - This is a positive pair that should be close in vector space.
- **Triplets:** `["anchor", "positive", "negative"]` - This is a triplet: The `positive` text should be close to the `anchor`, while the `negative` text should be distant to the `anchor`.
- **Sets:** `{"set": ["text1", "text2", ...]}` A set of texts describing the same thing, e.g. different paraphrases of the same question, different captions for the same image. Any combination of the elements is considered as a positive pair.
- **Query-Pairs:** `{"query": "text", "pos": ["text1", "text2", ...]}` A query together with a set of positive texts. Can be formed to a pair `["query", "positive"]` by randomly selecting a text from `pos`.
- **Query-Triplets:** `{"query": "text", "pos": ["text1", "text2", ...], "neg": ["text1", "text2", ...]}` A query together with a set of positive texts and negative texts. Can be formed to a triplet `["query", "positive", "negative"]` by randomly selecting a text from `pos` and `neg`.

## Available Datasets

**Note: I'm currently in the process to upload the files. Please check again next week to get the full list of datasets** 

We measure the performance for each training dataset by training the [nreimers/MiniLM-L6-H384-uncased](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) model on it with [MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss), a batch size of 256, for 2000 training steps. The performance is then averaged across 14 sentence embedding benchmark datasets from diverse domains (Reddit, Twitter, News, Publications, E-Mails, ...).



| Dataset | Description | Size (#Lines) | Performance | Reference |
| --- | --- | :---: | :---: | --- |
| [gooaq_pairs.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/gooaq_pairs.jsonl.gz) | (Question, Answer)-Pairs from Google auto suggest | 3,012,496 | 59.06 | [GooAQ](https://github.com/allenai/gooaq)
| [yahoo_answers_title_answer.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/yahoo_answers_title_answer.jsonl.gz) | (Title, Answer) pairs from Yahoo Answers | 1,198,260 | 58.65 | [Yahoo Answers](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset)
| [msmarco-triplets.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/msmarco-triplets.jsonl.gz) | (Question, Answer, Negative)-Triplets from MS MARCO Passages dataset | 499,184 | 58.76 | [MS MARCO Passages](https://github.com/microsoft/MSMARCO-Passage-Ranking)
| [stackexchange_duplicate_questions_title_title.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/stackexchange_duplicate_questions_title_title.jsonl.gz) | (Title, Title) pairs of duplicate questions from StackExchange | 304,525 | 58.47 | [Stack Exchange Data API](https://data.stackexchange.com/apple/query/fork/1456963)
| [eli5_question_answer.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/eli5_question_answer.jsonl.gz) | (Question, Answer)-Pairs from ELI5 dataset | 325,475 | 58.24 | [ELI5](https://huggingface.co/datasets/eli5)
| [yahoo_answers_title_question.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/yahoo_answers_title_question.jsonl.gz) | (Title, Question_Body) pairs from Yahoo Answers | 659,896 | 58.05 | [Yahoo Answers](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset)
| [squad_pairs.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/squad_pairs.jsonl.gz) |  (Question, Answer_Passage) Pairs from SQuAD dataset | 87,599 | 58.02 | [SQuAD](https://huggingface.co/datasets/squad)
| [yahoo_answers_question_answer.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/yahoo_answers_question_answer.jsonl.gz) | (Question_Body, Answer) pairs from Yahoo Answers | 681,164 | 57.74 | [Yahoo Answers](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset)
| [wikihow.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/wikihow.jsonl.gz) | (Summary, Text) from WikiHow | 128,542 | 57.67 | [WikiHow](https://github.com/pvl/wikihow_pairs_dataset)
| [amazon_review_2018.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/amazon_review_2018.jsonl.gz) | (Title, review) pairs from Amazon | 87,877,725 | 57.65 | [Amazon review data (2018)](http://deepyeti.ucsd.edu/jianmo/amazon/index.html)
| [NQ-train_pairs.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/NQ-train_pairs.jsonl.gz) | Training pairs (query, answer_passage) from the NQ dataset | 100,231 | 57.48 | [Natural Questions](https://ai.google.com/research/NaturalQuestions)
| [amazon-qa.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/amazon-qa.jsonl.gz) | (Question, Answer) pairs from Amazon | 1,095,290 | 57.48 | [AmazonQA](https://github.com/amazonqa/amazonqa)
| [S2ORC_title_abstract.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/S2ORC_title_abstract.jsonl.gz) | (Title, Abstract) pairs of scientific papers | 41,769,185 | 57.39 | [S2ORC](https://github.com/allenai/s2orc)
| [quora_duplicates.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/quora_duplicates.jsonl.gz) | Duplicate question pairs from Quora | 103,663 | 57.36 | [QQP](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)
| [WikiAnswers.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/WikiAnswers.jsonl.gz) | Sets of duplicates questions | 27,383,151 | 57.34 | [WikiAnswers Corpus](https://github.com/afader/oqa#wikianswers-corpus)
| [searchQA_top5_snippets.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/searchQA_top5_snippets.jsonl.gz) | Question + Top5 text snippets from SearchQA dataset. Top5 | 117,220 | 57.34 | [search_qa](https://huggingface.co/datasets/search_qa)
| [stackexchange_duplicate_questions_title-body_title-body.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/stackexchange_duplicate_questions_title-body_title-body.jsonl.gz) | (Title+Body, Title+Body) pairs of duplicate questions from StackExchange | 250,460 | 57.30 | [Stack Exchange Data API](https://data.stackexchange.com/apple/query/fork/1456963)
| [S2ORC_citations_titles.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/S2ORC_citations_titles.jsonl.gz) | Citation network (paper titles) | 51,030,086 | 57.28 | [S2ORC](https://github.com/allenai/s2orc)
| [stackexchange_duplicate_questions_body_body.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/stackexchange_duplicate_questions_body_body.jsonl.gz) | (Body, Body) pairs of duplicate questions from StackExchange | 250,519 | 57.26 | [Stack Exchange Data API](https://data.stackexchange.com/apple/query/fork/1456963)
| [agnews.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/agnews.jsonl.gz) | (Title, Description) pairs of news articles from the AG News dataset | 1,157,745 | 57.25 | [AG news corpus](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
| [quora_duplicates_triplets.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/quora_duplicates_triplets.jsonl.gz) | Duplicate question pairs from Quora with additional hard negatives (mined & denoised by cross-encoder) | 101,762 | 56.97 | [QQP](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)
| [AllNLI.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/AllNLI.jsonl.gz) | Combination of SNLI + MultiNLI Triplets: (Anchor, Entailment_Text, Contradiction_Text) | 277,230 | 56.57 | [SNLI](https://huggingface.co/datasets/snli) and [MNLI](https://huggingface.co/datasets/multi_nli)
| [npr.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/npr.jsonl.gz) | (Title, Body) pairs from the npr.org website | 594,384 | 56.44 | [Pushshift](https://files.pushshift.io/news/)
| [specter_train_triples.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/specter_train_triples.jsonl.gz) | Triplets (Title, related_title, hard_negative) for Scientific Publications from Specter | 684,100 | 56.32 | [SPECTER](https://github.com/allenai/specter)
| [SimpleWiki.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/SimpleWiki.jsonl.gz) | Matched pairs (English_Wikipedia, Simple_English_Wikipedia) | 102,225 | 56.15 | [SimpleWiki](https://cs.pomona.edu/~dkauchak/simplification/)
| [PAQ_pairs.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/PAQ_pairs.jsonl.gz) | Training pairs (query, answer_passage) from the PAQ dataset | 64,371,441 | 56.11 | [PAQ](https://github.com/facebookresearch/PAQ)
| [altlex.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/altlex.jsonl.gz) | Matched pairs (English_Wikipedia, Simple_English_Wikipedia) | 112,696 | 55.95 | [altlex](https://github.com/chridey/altlex/)
| [ccnews_title_text.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/ccnews_title_text.jsonl.gz) | (Title, article) pairs from the CC News dataset | 614,664 | 55.84 | [CC-News](https://huggingface.co/datasets/cc_news)
| [codesearchnet.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/codesearchnet.jsonl.gz) | CodeSearchNet corpus is a dataset of (comment, code) pairs from opensource libraries hosted on GitHub. It contains code and documentation for several programming languages. | 1,151,414 | 55.80 | [CodeSearchNet](https://huggingface.co/datasets/code_search_net)
| [S2ORC_citations_abstracts.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/S2ORC_citations_abstracts.jsonl.gz) | Citation network (paper abstracts) | 39,567,485 | 55.74 | [S2ORC](https://github.com/allenai/s2orc)
| [sentence-compression.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/sentence-compression.jsonl.gz) | Pairs (long_text, short_text) about sentence-compression | 180,000 | 55.63 | [Sentence-Compression](https://github.com/google-research-datasets/sentence-compression)
| [TriviaQA_pairs.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/TriviaQA_pairs.jsonl.gz) | Pairs (query, answer) from TriviaQA dataset | 73,346 | 55.56 | [TriviaQA](https://huggingface.co/datasets/trivia_qa)
| [cnn_dailymail_splitted.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/cnn_dailymail_splitted.jsonl.gz) | (article, highlight sentence) with individual highlight sentences for each news article | 311,971 | 55.36 | [CNN Dailymail Dataset](https://huggingface.co/datasets/cnn_dailymail)
| [cnn_dailymail.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/cnn_dailymail.jsonl.gz) | (highlight sentences, article) with all highlight sentences as one text for each news article | 311,971 | 55.27 | [CNN Dailymail Dataset](https://huggingface.co/datasets/cnn_dailymail)
| [flickr30k_captions.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/flickr30k_captions.jsonl.gz) | Different captions for the same image from the Flickr30k dataset | 31,783 | 54.68 | [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)
| [xsum.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/xsum.jsonl.gz) | (Summary, News Article) pairs from XSUM dataset | 226,711 | 53.86 | [xsum](https://huggingface.co/datasets/xsum)
| [coco_captions.jsonl.gz](https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/coco_captions.jsonl.gz) | Different captions for the same image | 82,783 | 53.77 | [COCO](https://cocodataset.org/)





**Disclaimer:** We only distribute these datasets in a specific format, but we do not vouch for their quality or fairness, or claim that you have license to use the dataset. It remains the user's responsibility to determine whether you as a user have permission to use the dataset under the dataset's license and to cite the right owner of the dataset. Please check the individual dataset webpages for the license agreements.

If you're a dataset owner and wish to update any part of it, or do not want your dataset to be included in this dataset collection, feel free to contact me.