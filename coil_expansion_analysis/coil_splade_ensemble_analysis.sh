for i in {1..9}; do 
echo $i
python /home/adityasv/COILv2/SparseRetrieval/COIL/data_helpers/msmarco-passage/score_to_marco.py \
  --score_file /bos/tmp15/adityasv/COILv2-vkeshava/interpolation_splade_coil/interpolation-$i-rank.txt

# Get the metrics
python /bos/tmp16/vkeshava/tools/anserini/tools/scripts/msmarco/msmarco_passage_eval.py \
        /bos/tmp16/vkeshava/tools/anserini/tools/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt \
        /bos/tmp15/adityasv/COILv2-vkeshava/interpolation_splade_coil/interpolation-$i-rank.txt.marco

python ../beir_get_metrics.py  \
      --data_dir /bos/tmp15/adityasv/DomainAdaptation/MSMARCO/msmarco/ \
      --predictions_file /bos/tmp15/adityasv/COILv2-vkeshava/interpolation_splade_coil/interpolation-$i-rank.txt \
      --result_dump_path /bos/tmp15/adityasv/COILv2-vkeshava/interpolation_splade_coil/interpolation-$i-rank.results.tsv
done
