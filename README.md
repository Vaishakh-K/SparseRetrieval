# SparseRetrieval

This repo contains information on different experiments conducted in the area of Sparse Information Retrieval using contextualised term weights.
We have used the [Anserini](https://github.com/castorini/anserini) IR toolkit to perform the experiments.

# Results
|  Term Weighting |   Expansion  | MRR@10 |                                         Notes                                        |
|:---------------:|:------------:|:------:|:------------------------------------------------------------------------------------:|
| BM25            | None         | 0.1874 |                                                                                      |
| DeepCT          | None         | 0.2362 |                               No hyperparameter tuning                               |
| DeepCT          | None         | 0.2425 |                         Hyper-params fine-tuned: k1=18, b=0.7                        |
| DeepImpact      | doc2query–T5 | 0.3253 |                                                                                      |
| COIL-tok        | None         | 0.3536 |                                token dimension d = 32                                |
| uniCOIL         | doc2query–T5 | 0.3515 |                                                                                      |