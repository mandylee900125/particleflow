
cd particleflow/test_tmp_delphes

#generate pytorch data files from pkl files
python3 ../particleflow/mlpf/pytorch/graph_data_delphes.py --dataset data/pythia8_ttbar \
  --processed_dir data/pythia8_ttbar/processed --num-files-merge 1 --num-proc 1

  #generate pytorch data files from pkl files
  python3 ../particleflow/mlpf/pytorch/graph_data_delphes.py --dataset data/pythia8_qcd \
    --processed_dir data/pythia8_qcd/processed --num-files-merge 1 --num-proc 1
