#Model ablation experiment

python emb_tools/creatlmdbbyesme.py \
  -m esm2_15b \
  -f dataset/Strings_plant50/protein.fasta \
  -o emb/Strings_plant50_bf16 \
  --suffix Strings_plant50_bf16.avg.1280 \
  --commit-interval 100 --precision bf16 \
  --input-layer-ckpt results/Strings_plant50_bf16/esm2_15b/sepal-ppi-fusion_finetune_avg_to_1280/input_layer.pth

#nonefeature
python sepal-ppi.py --hydra --config Ablation_experiment/Strings_plant50bf16/nonefeature --epochs 10 \
  --output-dir results/Strings_plant50_bf16/esm2_15b/nonefeature

python sepal-ppi.py --mode ensemble_train --ensemble-config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/ensemble_nonefeature.yaml \
  --predict-prot-feature-folder mutifeature/Strings_plant50 --output-dir results/Strings_plant50_bf16/esm2_15b/cis_ensemble_nonefeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/nonefeature/predict_Gly_hh70.yaml  \
  --output-dir results/Strings_plant50_bf16_nonefeature/gly/hh70 \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/nonefeature/predict_Gly_hh50.yaml  \
  --output-dir results/Strings_plant50_bf16_nonefeature/gly/hh50 \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/nonefeature/predict_Gly_hl.yaml  \
  --output-dir results/Strings_plant50_bf16_nonefeature/gly/hl \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/nonefeature/predict_Gly_ll.yaml  \
  --output-dir results/Strings_plant50_bf16_nonefeature/gly/ll \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

#noneatt
python sepal-ppi.py --hydra --config Ablation_experiment/Strings_plant50bf16/noneatt --epochs 10 \
  --output-dir results/Strings_plant50_bf16/esm2_15b/noneatt

python sepal-ppi.py --mode ensemble_train --ensemble-config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/ensemble_noneatt.yaml \
  --predict-prot-feature-folder mutifeature/Strings_plant50 --output-dir results/Strings_plant50_bf16/esm2_15b/cis_ensemble_noneatt

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/noneatt/predict_Gly_hh70.yaml  \
  --output-dir results/Strings_plant50_bf16_noneatt/gly/hh70 \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/noneatt/predict_Gly_hh50.yaml  \
  --output-dir results/Strings_plant50_bf16_noneatt/gly/hh50 \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/noneatt/predict_Gly_hl.yaml  \
  --output-dir results/Strings_plant50_bf16_noneatt/gly/hl \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/noneatt/predict_Gly_ll.yaml  \
  --output-dir results/Strings_plant50_bf16_noneatt/gly/ll \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

#noneProjector
python sepal-ppi.py --hydra --config Ablation_experiment/Strings_plant50bf16/noneProjector --epochs 10 \
  --output-dir results/Strings_plant50_bf16/esm2_15b/noneProjector

python sepal-ppi.py --mode ensemble_train --ensemble-config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/ensemble_noneProjector.yaml \
  --predict-prot-feature-folder mutifeature/Strings_plant50 --output-dir results/Strings_plant50_bf16/esm2_15b/cis_ensemble_noneProjector

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/noneProjector/predict_Gly_hh70.yaml  \
  --output-dir results/Strings_plant50_bf16_noneProjector/gly/hh70 \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/noneProjector/predict_Gly_hh50.yaml  \
  --output-dir results/Strings_plant50_bf16_noneProjector/gly/hh50 \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/noneProjector/predict_Gly_hl.yaml  \
  --output-dir results/Strings_plant50_bf16_noneProjector/gly/hl \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/sepal-ppi/Ablation_experiment/Strings_plant50bf16/noneProjector/predict_Gly_ll.yaml  \
  --output-dir results/Strings_plant50_bf16_noneProjector/gly/ll \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature