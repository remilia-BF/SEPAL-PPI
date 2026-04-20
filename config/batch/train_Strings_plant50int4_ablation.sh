#Model ablation experiment

python emb_tools/creatlmdbbyesme.py \
  -m esm2_15b \
  -f dataset/Strings_plant50/protein.fasta \
  -o emb/Strings_plant50_int4 \
  --suffix Strings_plant50_int4.avg.1280 \
  --commit-interval 100 --precision int4 \
  --input-layer-ckpt results/Strings_plant50_int4/esm2_15b/sepal-ppi-fusion_finetune_avg_to_1280/input_layer.pth

#noneatt
python sepal-ppi.py --hydra --config Ablation_experiment/Strings_plant50int4/noneatt --epochs 10 \
  --output-dir results/Strings_plant50_int4/esm2_15b/noneatt

python sepal-ppi.py --mode ensemble_train --ensemble-config config/sepal-ppi/Ablation_experiment/Strings_plant50int4/ensemble_noneatt.yaml \
  --predict-prot-feature-folder mutifeature/Strings_plant50 --output-dir results/Strings_plant50_int4/esm2_15b/cis_ensemble_noneatt

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50int4/noneatt/predict_Gly_hh70.yaml  \
  --output-dir results/Strings_plant50_int4/gly/hh70 \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50int4/noneatt/predict_Gly_hh50.yaml  \
  --output-dir results/Strings_plant50_int4/gly/hh50 \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50int4/noneatt/predict_Gly_hl.yaml  \
  --output-dir results/Strings_plant50_int4/gly/hl \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50int4/noneatt/predict_Gly_ll.yaml  \
  --output-dir results/Strings_plant50_int4/gly/ll \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

#nonefeature
python sepal-ppi.py --hydra --config Ablation_experiment/Strings_plant50int4/nonefeature --epochs 10 \
  --output-dir results/Strings_plant50_int4/esm2_15b/nonefeature

python sepal-ppi.py --mode ensemble_train --ensemble-config config/sepal-ppi/Ablation_experiment/Strings_plant50int4/ensemble_nonefeature.yaml \
  --predict-prot-feature-folder mutifeature/Strings_plant50 --output-dir results/Strings_plant50_int4/esm2_15b/cis_ensemble_nonefeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50int4/nonefeature/predict_Gly_hh70.yaml  \
  --output-dir results/Strings_plant50_int4/gly/hh70 \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50int4/nonefeature/predict_Gly_hh50.yaml  \
  --output-dir results/Strings_plant50_int4/gly/hh50 \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50int4/nonefeature/predict_Gly_hl.yaml  \
  --output-dir results/Strings_plant50_int4/gly/hl \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50int4/nonefeature/predict_Gly_ll.yaml  \
  --output-dir results/Strings_plant50_int4/gly/ll \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature