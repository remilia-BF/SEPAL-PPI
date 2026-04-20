

#feature消融 - noHMM

python sepal-ppi.py --mode ensemble_predict --config config/predict/bf16-feature-ablation/predict_Gly_hh70_noHMM.yaml  \
  --output-dir results/Strings_plant50_bf16/gly/hh70_noHMM \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/predict/bf16-feature-ablation/predict_Gly_hh50_noHMM.yaml  \
  --output-dir results/Strings_plant50_bf16/gly/hh50_noHMM \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/predict/bf16-feature-ablation/predict_Gly_hl_noHMM.yaml  \
  --output-dir results/Strings_plant50_bf16/gly/hl_noHMM \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/predict/bf16-feature-ablation/predict_Gly_ll_noHMM.yaml  \
  --output-dir results/Strings_plant50_bf16/gly/ll_noHMM \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

#feature消融 - nosasa

python sepal-ppi.py --mode ensemble_predict --config config/predict/bf16-feature-ablation/predict_Gly_hh70_nosasa.yaml  \
  --output-dir results/Strings_plant50_bf16/gly/hh70_nosasa \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/predict/bf16-feature-ablation/predict_Gly_hh50_nosasa.yaml  \
  --output-dir results/Strings_plant50_bf16/gly/hh50_nosasa \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/predict/bf16-feature-ablation/predict_Gly_hl_nosasa.yaml  \
  --output-dir results/Strings_plant50_bf16/gly/hl_nosasa \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/predict/bf16-feature-ablation/predict_Gly_ll_nosasa.yaml  \
  --output-dir results/Strings_plant50_bf16/gly/ll_nosasa \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

#feature消融 - nosst (no secondary structure)

python sepal-ppi.py --mode ensemble_predict --config config/predict/bf16-feature-ablation/predict_Gly_hh70_nosst.yaml  \
  --output-dir results/Strings_plant50_bf16/gly/hh70_nosst \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/predict/bf16-feature-ablation/predict_Gly_hh50_nosst.yaml  \
  --output-dir results/Strings_plant50_bf16/gly/hh50_nosst \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/predict/bf16-feature-ablation/predict_Gly_hl_nosst.yaml  \
  --output-dir results/Strings_plant50_bf16/gly/hl_nosst \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/predict/bf16-feature-ablation/predict_Gly_ll_nosst.yaml  \
  --output-dir results/Strings_plant50_bf16/gly/ll_nosst \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature