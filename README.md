# SEPAL-PPI

**Structure-Enhanced Prediction with Attention-based Learning for Protein-Protein Interactions**

<img width="1920" height="1080" alt="FIGURE1" src="https://github.com/user-attachments/assets/f77d1555-c84d-4d66-9da8-8ef570c0eef7" />



SEPAL-PPI is a research codebase for protein-protein interaction prediction that combines protein language model embeddings, multimodal structural features, and ensemble inference. This repository is intended for full training, feature generation, benchmark reproduction, and research-oriented prediction workflows.

## Quickly start inferring PPI at the plant proteome level

If you only need prediction on new sequences or a small custom dataset, the recommended workflow is the standalone pip package **`onestep-sepal-ppi`**, rather than the full research repository.

Example setup:

```bash
conda create -n sepal python=3.12 -y
conda activate sepal

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-build-isolation flash-attn
pip install esm-efficient
pip install onestep-sepal-ppi
```

Then initialize the runtime config and run one-step prediction:

```bash
sepal-ppi setup --ini-path /path/to/nsp3_env_config.ini

sepal-ppi prepare \
  -f /path/to/input.fasta \
  --output-dir /path/to/mutifeature

sepal-ppi predict \
  --esm-precision bf16 \
  --esm-model esm2_15b \
  --multifeature-dir /path/to/mutifeature \
  --interaction-list /path/to/pairs.csv \
  --fasta /path/to/input.fasta \
  --output-dir results/simple-predict
```

---
## For the installation of a complete framework project

## Installation

### 1. Create the conda environment

```bash
conda create -n SEPAL-PPI python=3.12 -y
conda activate SEPAL-PPI
```

### 2. Install PyTorch first

Install the CUDA-matched PyTorch stack before the repository requirements. For the environment used in this project:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install repository dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional packages

Install the following only if your workflow needs them:

- `dgl`: graph-based experiments used with the matching CUDA channel
- `torch_geometric`: graph-based experiments used with the matching wheel index
- `flash-attn`: faster attention kernels on supported CUDA systems
- `faiss-cpu` or `faiss-gpu`: CIS interpretability / nearest-neighbor search
- `pdb2pqr`, `propka`, `mmcif-pdbx`: structure-based feature generation utilities



## Full prediction workflow in this repository

### 1. Prepare multimodal features

```bash
python mutifeature_tools/one_step_mutifeature.py \
  -p dataset/<your_dataset>/pdb \
  -f dataset/<your_dataset>/protein.fasta \
  -o mutifeature/<your_dataset>
```

**Note:** ProSST JSON files are not distributed in this repository. The original ProSST project uses a CC BY-NC-ND 4.0 license, so the related generation scripts are not bundled here for redistribution.

### 2. Generate embeddings

```bash
python emb_tools/creatlmdb.py \
  -m esm2_15b \
  -f dataset/<your_dataset>/protein.fasta \
  -o emb/<your_dataset> \
  --suffix <your_dataset> \
  --commit-interval 50 \
  --input-layer-ckpt results/final/Human/esm15b/sepal-ppi-avg_change640-nofeature-avg/input_layer.pth
```

### 3. Run ensemble prediction

Edit `config/predict/<your_dataset>_esm2_15b.yaml`, then run:

```bash
python sepal-ppi.py \
  --mode ensemble_predict \
  --config config/predict/<your_dataset>_esm2_15b.yaml \
  --output-dir results/predict/<your_dataset> \
  --predict-prot-feature-folder mutifeature/<your_dataset>
```

### 4. View results

```bash
cd results/predict/<your_dataset>
python -m http.server 2345
```

Open `http://localhost:2345/prediction_results.html` to inspect prediction tables, ensemble scores, interpretability outputs, and downloadable CSV files.

## Reproducing the paper workflow

Example for the Human benchmark:

```bash
python sepal-ppi.py \
  --mode ensemble_predict \
  --config config/predict/Human_esm2_15b.yaml \
  --output-dir results/final/Human/esm2_15b/ensemble_predict-Human_predict_feature \
  --predict-prot-feature-folder mutifeature/Human
```

Reproducing the paper workflow
To reproduce all training runs reported in the manuscript, pre-written batch scripts are provided under config/batch/. Simply run the script corresponding to the dataset/configuration you need:
bash# Available scripts
ls config/batch/
# train_Human.sh
# train_Interact_Ara.sh
# train_Strings_plant50bf16.sh
# train_Strings_plant50int4.sh
# train_Strings_plant50int8.sh

# Example: reproduce the Strings_plant50 int4 run
bash config/batch/train_Strings_plant50int4.sh

# Example: reproduce the Human benchmark
bash config/batch/train_Human.sh
Each script runs the complete pipeline end-to-end (embedding generation → base model → fusion finetune → residue-level model → CIS → ensemble) without further manual intervention.

Training on your own data
The full training pipeline consists of seven sequential steps. The example below uses <your_dataset> as a placeholder — substitute your actual dataset name and paths throughout.
Step 0: Prepare multimodal features
```bash
python mutifeature_tools/one_step_mutifeature.py \
  -p dataset/<your_dataset>/pdb \
  -f dataset/<your_dataset>/protein.fasta \
  -o mutifeature/<your_dataset>
```
Step 1: Generate raw embeddings
```bash
python emb_tools/creatlmdbbyesme.py \
  -m esm2_15b \
  -f dataset/<your_dataset>/protein.fasta \
  -o emb/<your_dataset> \
  --suffix <your_dataset> \
  --commit-interval 100 \
  --precision <int4|int8|bf16> \
  --no-bucket
```
Step 2: Train the base model (Average Pooling)
```bash
python sepal-ppi.py \
  --hydra \
  --config <your_dataset>/sepal-ppi-avg-noinput \
  --epochs 25 \
  --output-dir results/<your_dataset>/esm2_15b/sepal-ppi-avg-noinput
```
Step 3: Fusion finetune to obtain the input projection layer
```bash
python sepal-ppi.py \
  --hydra \
  --config <your_dataset>/fusion_finetune_avg_to_1280 \
  --mode fusion_finetune \
  --output-dir results/<your_dataset>/esm2_15b/sepal-ppi-fusion_finetune_avg_to_1280
```
Step 4: Generate projected residue-level embeddings
```bash
python emb_tools/creatlmdbbyesme.py \
  -m esm2_15b \
  -f dataset/<your_dataset>/protein.fasta \
  -o emb/<your_dataset> \
  --suffix <your_dataset>.avg.1280 \
  --commit-interval 100 \
  --precision <int4|int8|bf16> \
  --input-layer-ckpt results/<your_dataset>/esm2_15b/sepal-ppi-fusion_finetune_avg_to_1280/input_layer.pth \
  --source-lmdb emb/<your_dataset>/esm2_15b.<your_dataset>.lmdb
```
Step 5: Train the residue-level model (Projector Feature Contant)
```bash
python sepal-ppi.py \
  --hydra \
  --config <your_dataset>/sepal-ppi-projector-feature-contant \
  --epochs 10 \
  --output-dir results/<your_dataset>/esm2_15b/sepal-ppi-projector-feature-contant
```
Note: After this step you may remove the intermediate raw embedding LMDB to save disk space:
bashrm emb/<your_dataset>/esm2_15b.<your_dataset>.lmdb/noCLSeos.lmdb

Step 6: Train the CIS model
```bash
python sepal-ppi.py \
  --hydra \
  --config <your_dataset>/sepal-ppi-cis \
  --epochs 50 \
  --output-dir results/<your_dataset>/esm2_15b/sepal-ppi-cis
```
Step 7: Train the ensemble
```bash
python sepal-ppi.py \
  --mode ensemble_train \
  --ensemble-config config/sepal-ppi/<your_dataset>/esm_ensemble.yaml \
  --predict-prot-feature-folder mutifeature/<your_dataset> \
  --output-dir results/<your_dataset>/esm2_15b/cis_residue_ensemble
```
Tip: For a fully worked concrete example of each step above, refer to config/batch/train_Strings_plant50int4.sh. The paper-reproduction scripts in config/batch/ follow exactly this seven-step structure with dataset-specific paths and hyperparameters already filled in.

## Repository layout

```text
SEPAL-PPI/
├── sepal-ppi.py           # Main training / prediction entry point
├── config/                # Hydra and model configuration files
├── dataset/               # Paper datasets and release-ready metadata
├── emb_tools/             # Embedding generation scripts
├── mutifeature_tools/     # Multimodal feature generation scripts
├── predict/               # Prediction examples and release data
├── src/                   # Core implementation
└── requirements.txt       # Base Python dependencies
```

## Citation

If you use SEPAL-PPI in your research, please cite the accompanying paper or preprint associated with this repository.

## License 

MIT License

Copyright (c) 2026 Yingjie Zhang, et al.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## IMPORTANT NOTICE REGARDING THIRD-PARTY DEPENDENCIES

This repository contains the original source code for the SEPAL framework, 
which is licensed under the permissive MIT License above and is suitable for 
commercial use.

HOWEVER, to facilitate training and inference as described in the manuscript, 
SEPAL relies on external resources and pre-trained models that are NOT 
included in this repository due to their upstream license restrictions. 

Users are solely responsible for obtaining these resources from their original 
sources and for ensuring their usage complies with the respective licenses.

Specifically, the following components are NOT distributed in this codebase:

1.  **S-PLM (Structural Protein Language Model)**
    - **Status:** NOT INCLUDED.
    - **Reason:** The S-PLM resources are distributed under a Creative Commons 
      (CC) license (e.g., CC BY-NC-SA 4.0 or similar variants) which imposes 
      specific restrictions regarding commercial use and/or share-alike terms.
    - **Action Required:** Users must download S-PLM weights/embeddings directly 
      from the official S-PLM repository or source and adhere to its specific 
      CC license terms.

2.  **NetSurfP-3.0**
    - **Status:** NOT INCLUDED.
    - **Reason:** NetSurfP-3.0 is released under a license strictly limited to 
      **Academic Use Only**. Commercial usage, or redistribution within a 
      commercially licensed software package (like MIT), is prohibited by the 
      upstream authors.
    - **Action Required:** Users affiliated with academic or non-commercial 
      institutions must download NetSurfP-3.0 from its official DTU Health Tech 
      services. Commercial entities or users seeking to deploy SEPAL in a 
      commercial setting MUST NOT use NetSurfP-3.0 and should seek alternative 
      methods for structural feature generation or contact the NetSurfP authors 
      for a potential commercial license.

**Disclaimer:** The inclusion of interfaces or reference calls to these tools in 
the SEPAL source code does not imply any endorsement or re-licensing of those 
third-party components. The user assumes all liability for ensuring compliance 
with the license terms of S-PLM and NetSurfP-3.0.
