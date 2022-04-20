import os
import argparse
import pytrec_eval
from beir.retrieval.custom_metrics import mrr
from pprint import pprint
from typing import Type, List, Dict, Union, Tuple


def evaluate(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    
        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        _mrr = {}
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0
        
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)
        
        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
        

        _mrr = mrr(qrels, results, k_values)

        for eval in [ndcg, _map, recall, precision, _mrr]:
            for k in eval.keys():
                print("{}: {:.4f}".format(k, eval[k]))

        return ndcg, _map, recall, precision, _mrr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help='path to the dataset')
    parser.add_argument("--predictions_file", type=str, required=True, help='path to the predictions TSV file. Schema: qid doc_id score')
    parser.add_argument("--result_dump_path", type=str, required=True, help='path dump the scores')
    parser.add_argument("--k_values", type=str, default="10,1000", help='comma separated k values for Metric@k measurement')
    
    args = parser.parse_args()

    args.k_values = list(map(int, args.k_values.strip().split(',')))


    qrels_path = os.path.join(args.data_dir, "qrels")
    if "msmarco" in args.data_dir:
        qrels_path = os.path.join(qrels_path, "dev.tsv")
    else:
        qrels_path = os.path.join(qrels_path, "test.tsv")

    qrels = {}
    with open(qrels_path, 'r') as fi:
        for idx, line in enumerate(fi):
            if idx == 0:
                continue

            query_id, doc_id, score = line.strip().split('\t')
            score = int(score)

            if query_id not in qrels:
                qrels[query_id] = {doc_id: score}
            else:
                qrels[query_id][doc_id] = score

    retrieval_results = {}
    with open(args.predictions_file, 'r') as fi:
        for line in fi: 
            query_id, doc_id, score = line.strip().split('\t')
            score = float(score)

            if query_id not in retrieval_results:
                retrieval_results[query_id] = {doc_id: score}
            else:
                retrieval_results[query_id][doc_id] = score
    
    results = evaluate(qrels, retrieval_results, args.k_values)

    pprint(results)
    with open(args.result_dump_path, 'w') as fo:
        fo.write(f"METRIC\tVALUE\n")
        for result in results:
            for k,v in result.items():
                fo.write(f"{k}\t{v}\n")

    not_retrieved = []
    if "msmarco" in args.data_dir:
        for query_id, retrieved_documents in retrieval_results.items():
            for rel_doc in qrels[query_id].keys():
                if rel_doc not in retrieved_documents:
                    not_retrieved.append((query_id, rel_doc))

    print('not retrieved', len(not_retrieved))

    out_path = os.path.join(os.path.dirname(args.result_dump_path), "not_retrieved_docs.tsv")
    with open(out_path, 'w') as fo:
        out_line = "\t".join(["query_id", "rel_doc_id"]) + "\n"
        fo.write(out_line)
        for query_id, rel_doc in not_retrieved:
            out_line = "\t".join([query_id, rel_doc]) + "\n"
            fo.write(out_line)

if __name__=="__main__":
    main()