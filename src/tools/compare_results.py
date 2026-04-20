
import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import seaborn as sns
import pandas as pd
from collections import defaultdict
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Compare machine learning results from multiple folders.")
    parser.add_argument("--analyze_folder", nargs='+', required=True, help="List of result folders to analyze")
    parser.add_argument("--dataset", required=True, help="Path to dataset folder containing c2Validation.txt, c3Test.txt and protein.fasta")
    parser.add_argument("--output_dir", default="comparison_results", help="Directory to save output plots and lists")
    return parser.parse_args()

def load_sequences(dataset_path):
    fasta_path = os.path.join(dataset_path, "protein.fasta")
    seqs = {}
    if not os.path.exists(fasta_path):
        print(f"Warning: Fasta file not found at {fasta_path}. Sequences will be empty.")
        return seqs
    
    current_id = None
    current_seq = []
    try:
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if current_id:
                        seqs[current_id] = "".join(current_seq)
                    # Parse ID: assuming typical fasta header, take first token
                    # Adjust if needed based on header format e.g. ">Q9FHS0 ..." -> "Q9FHS0"
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id:
                seqs[current_id] = "".join(current_seq)
        print(f"Loaded {len(seqs)} sequences from {fasta_path}")
    except Exception as e:
        print(f"Error reading fasta file: {e}")
        
    return seqs

def load_data(folder_path):
    json_path = os.path.join(folder_path, "final_evaluation_summary.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Structure: {'c2': {'protein_ids': [...]}, 'c3': ...}
    return data

def get_predictions_df(data, dataset_key):
    if dataset_key not in data:
        return None
    
    entries = data[dataset_key].get('protein_ids', [])
    if not entries:
        return None
    
    df = pd.DataFrame(entries)
    # Ensure columns exist
    required_cols = ['protein1_id', 'protein2_id', 'probability', 'logit', 'label']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column {col} missing in data.")
            return None
            
    # Create unique ID
    df['pair_id'] = df.apply(lambda x: f"{x['protein1_id']}_{x['protein2_id']}", axis=1)
    return df

def plot_sorted_probabilities(dfs, labels, output_dir, dataset_name):
    plt.figure(figsize=(10, 6))
    for df, label in zip(dfs, labels):
        probs = df['probability'].sort_values(ascending=False).values
        plt.plot(probs, label=label)
    
    plt.xlabel('Rank')
    plt.ylabel('Prediction Probability')
    plt.title(f'Sorted Prediction Probabilities ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_sorted_probs.png'))
    plt.close()

def plot_logit_distribution(dfs, labels, output_dir, dataset_name):
    plt.figure(figsize=(10, 6))
    for df, label in zip(dfs, labels):
        sns.kdeplot(df['logit'], label=label, fill=True, alpha=0.3)
    
    plt.xlabel('Logit')
    plt.ylabel('Density')
    plt.title(f'Logit Distribution ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_logit_dist.png'))
    plt.close()

def plot_aupr(dfs, labels, output_dir, dataset_name):
    plt.figure(figsize=(10, 6))
    for df, label in zip(dfs, labels):
        y_true = df['label']
        y_scores = df['probability']
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        aupr = auc(recall, precision)
        plt.plot(recall, precision, label=f'{label} (AUPR = {aupr:.4f})')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_aupr.png'))
    plt.close()

def save_detailed_comparison(pair_ids, dfs, labels, seq_dict, filepath):
    """
    Save detailed information for a set of pair_ids into a CSV/TSV file.
    Includes: P1, P2, Sequences, True Label, and for each model: Prediction, Probability, Logit.
    """
    data_rows = []
    
    # Pre-index DataFrames for faster lookup
    df_lookups = [df.set_index('pair_id') for df in dfs]
    
    # Use the first dataframe as the source for shared logic (P1, P2, Label)
    base_lookup = df_lookups[0]
    
    for pid in pair_ids:
        if pid not in base_lookup.index:
            continue
            
        base_row = base_lookup.loc[pid]
        # Handle duplicate indices
        if isinstance(base_row, pd.DataFrame):
            base_row = base_row.iloc[0]
            
        p1 = base_row['protein1_id']
        p2 = base_row['protein2_id']
        true_label = base_row['label']
        
        row_data = {
            'Protein1': p1,
            'Protein2': p2,
            'Protein1_Sequence': seq_dict.get(p1, "Sequence Not Found"),
            'Protein2_Sequence': seq_dict.get(p2, "Sequence Not Found"),
            'True_Label': true_label
        }
        
        for i, (df_map, model_name) in enumerate(zip(df_lookups, labels)):
            if pid in df_map.index:
                m_row = df_map.loc[pid]
                if isinstance(m_row, pd.DataFrame):
                    m_row = m_row.iloc[0]
                
                prob = float(m_row['probability'])
                logit = float(m_row['logit'])
                pred = 1 if prob >= 0.5 else 0
                
                row_data[f'{model_name}_Prediction'] = pred
                row_data[f'{model_name}_Probability'] = prob
                row_data[f'{model_name}_Logit'] = logit
            else:
                row_data[f'{model_name}_Prediction'] = None
                row_data[f'{model_name}_Probability'] = None
                row_data[f'{model_name}_Logit'] = None
        
        data_rows.append(row_data)
        
    if data_rows:
        res_df = pd.DataFrame(data_rows)
        # Using comma separator
        res_df.to_csv(filepath, index=False)
    else:
        print(f"Warning: No data to save for {filepath}")

def plot_probability_distribution(dfs, labels, output_dir, dataset_name):
    plt.figure(figsize=(10, 6))
    for df, label in zip(dfs, labels):
        # Clip probabilities to [0,1] just in case
        sns.kdeplot(df['probability'], label=label, fill=True, alpha=0.3, clip=(0, 1))
    
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title(f'Prediction Probability Distribution ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_prob_dist.png'))
    plt.close()

def analyze_top_k_metrics(dfs, labels, output_dir, dataset_name, max_k=500):
    k_values = list(range(1, max_k + 1))
    metrics_data = []
    
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Precision
    plt.subplot(1, 2, 1)
    
    # Store results for plotting
    plot_data_precision = {label: [] for label in labels}
    plot_data_recall = {label: [] for label in labels}
    plot_ks = {label: [] for label in labels}

    for df, label in zip(dfs, labels):
        df_sorted = df.sort_values(by='probability', ascending=False)
        total_positives = df_sorted['label'].sum()
        
        # Calculate for all K in one pass if possible, or just loop efficiently
        # Since K is small (up to 500), loop is fine.
        # But for cumulative sum it's faster.
        
        # Optimized calculation using cumulative sum
        top_max_k = df_sorted.head(max_k).copy()
        top_max_k['is_pos'] = top_max_k['label']
        cum_tp = top_max_k['is_pos'].cumsum().values
        
        for i, k in enumerate(k_values):
            # i corresponds to k-1
            if i < len(cum_tp):
                tp = cum_tp[i]
                actual_k = k
            else:
                # If dataset is smaller than k
                tp = cum_tp[-1]
                actual_k = len(cum_tp)
            
            precision = tp / actual_k if actual_k > 0 else 0
            recall = tp / total_positives if total_positives > 0 else 0
            
            metrics_data.append({
                'Model': label,
                'K': k,
                'Actual_K': actual_k,
                'Precision': precision,
                'Recall': recall,
                'TP': tp,
                'Total_Positives': total_positives
            })
            
            plot_data_precision[label].append(precision)
            plot_data_recall[label].append(recall)
            plot_ks[label].append(k)

    for label in labels:
        plt.plot(plot_ks[label], plot_data_precision[label], label=label)
    
    plt.xlabel('K')
    plt.ylabel('Precision')
    plt.title(f'Top-K Precision ({dataset_name})')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Recall
    plt.subplot(1, 2, 2)
    for label in labels:
        plt.plot(plot_ks[label], plot_data_recall[label], label=label)
        
    plt.xlabel('K')
    plt.ylabel('Recall')
    plt.title(f'Top-K Recall ({dataset_name})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_top_k_metrics.png'))
    plt.close()
    
    # Save CSV
    pd.DataFrame(metrics_data).to_csv(os.path.join(output_dir, f'{dataset_name}_top_k_metrics.csv'), index=False)

def analyze_top_percentile_metrics(dfs, labels, output_dir, dataset_name, start_p=0.1, end_p=5.0, step_p=0.01):
    metrics_data = []
    
    # Generate percentiles
    # Use linspace or integers to avoid floating point issues
    num_steps = int(round((end_p - start_p) / step_p)) + 1
    percentiles = [round(start_p + i * step_p, 4) for i in range(num_steps)]
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plot_data_precision = {label: [] for label in labels}
    plot_data_recall = {label: [] for label in labels}

    for df, label in zip(dfs, labels):
        df_sorted = df.sort_values(by='probability', ascending=False)
        total_positives = df_sorted['label'].sum()
        N = len(df_sorted)
        
        # Calculate max K needed (for the largest percentile)
        max_k = int(N * percentiles[-1] / 100)
        # Ensure at least 1 if N > 0, otherwise 0
        if max_k == 0 and N > 0: max_k = 1
        
        # Optimization: use cumsum
        if max_k > 0:
            top_max_k = df_sorted.head(max_k).copy()
            top_max_k['is_pos'] = top_max_k['label']
            cum_tp = top_max_k['is_pos'].cumsum().values
        else:
            cum_tp = []

        for p in percentiles:
            k = int(N * p / 100)
            if k == 0:
                k = 1
            
            # Lookup TP
            # k is 1-based size. Index is k-1.
            if k <= len(cum_tp):
                tp = cum_tp[k-1]
            elif len(cum_tp) > 0:
                tp = cum_tp[-1]
            else:
                tp = 0
            
            precision = tp / k if k > 0 else 0
            recall = tp / total_positives if total_positives > 0 else 0
            
            metrics_data.append({
                'Model': label,
                'Percentile': p,
                'K': k,
                'Precision': precision,
                'Recall': recall,
                'TP': tp
            })
            
            plot_data_precision[label].append(precision)
            plot_data_recall[label].append(recall)
            
    for label in labels:
        plt.plot(percentiles, plot_data_precision[label], label=label)
        
    plt.xlabel('Top Percentile (%)')
    plt.ylabel('Precision')
    plt.title(f'Top-% Precision ({dataset_name})')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for label in labels:
        plt.plot(percentiles, plot_data_recall[label], label=label)
        
    plt.xlabel('Top Percentile (%)')
    plt.ylabel('Recall')
    plt.title(f'Top-% Recall ({dataset_name})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_top_percentile_metrics.png'))
    plt.close()
    
    pd.DataFrame(metrics_data).to_csv(os.path.join(output_dir, f'{dataset_name}_top_percentile_metrics.csv'), index=False)

def perform_venn_analysis(dfs, labels, output_dir, dataset_name, seq_dict):
    if len(dfs) not in [2, 3]:
        print(f"Venn diagrams currently support 2 or 3 models. skipping for {len(dfs)} models.")
        return

    # Prepare Sets
    # 1. Predicted Positive (Threshold 0.5)
    pred_sets = []
    for df in dfs:
        pred_sets.append(set(df[df['probability'] >= 0.5]['pair_id']))
    
    # 2. Correct Predictions (TP + TN)
    correct_sets = []
    for df in dfs:
        is_correct = (df['probability'] >= 0.5) == (df['label'] == 1)
        correct_sets.append(set(df[is_correct]['pair_id']))

    # 3. Error Predictions (FP + FN)
    error_sets = []
    for df in dfs:
        is_error = (df['probability'] >= 0.5) != (df['label'] == 1)
        error_sets.append(set(df[is_error]['pair_id']))

    # Helper to plot and save
    def plot_and_save_venn(sets, labels, title, prefix):
        plt.figure(figsize=(8, 8))
        if len(sets) == 2:
            v = venn2(sets, set_labels=labels)
            set_a, set_b = sets[0], sets[1]
            ONLY_A = set_a - set_b
            ONLY_B = set_b - set_a
            COMMON = set_a & set_b
            
            save_detailed_comparison(ONLY_A, dfs, labels, seq_dict, 
                                     os.path.join(output_dir, f"{prefix}_only_{labels[0]}.csv"))
            save_detailed_comparison(ONLY_B, dfs, labels, seq_dict, 
                                     os.path.join(output_dir, f"{prefix}_only_{labels[1]}.csv"))
            save_detailed_comparison(COMMON, dfs, labels, seq_dict, 
                                     os.path.join(output_dir, f"{prefix}_common.csv"))
            
        elif len(sets) == 3:
            v = venn3(sets, set_labels=labels)
            try:
                # venn3 regions: 100, 010, 001, 110, 101, 011, 111
                # We can calculate explicitly using sets
                sets_dict = {
                    '100': sets[0] - sets[1] - sets[2],
                    '010': sets[1] - sets[0] - sets[2],
                    '001': sets[2] - sets[0] - sets[1],
                    '110': (sets[0] & sets[1]) - sets[2],
                    '101': (sets[0] & sets[2]) - sets[1],
                    '011': (sets[1] & sets[2]) - sets[0],
                    '111': sets[0] & sets[1] & sets[2]
                }
                
                # Save just the most relevant ones to avoid clutter (e.g. exclusive and all-common)
                # Or save all? Let's save Main ones.
                save_detailed_comparison(sets_dict['111'], dfs, labels, seq_dict,
                                         os.path.join(output_dir, f"{prefix}_common_all.csv"))
                
                save_detailed_comparison(sets_dict['100'], dfs, labels, seq_dict,
                                         os.path.join(output_dir, f"{prefix}_only_{labels[0]}.csv"))
                save_detailed_comparison(sets_dict['010'], dfs, labels, seq_dict,
                                         os.path.join(output_dir, f"{prefix}_only_{labels[1]}.csv"))
                save_detailed_comparison(sets_dict['001'], dfs, labels, seq_dict,
                                         os.path.join(output_dir, f"{prefix}_only_{labels[2]}.csv"))
                
            except Exception as e:
                print(f"Error saving venn 3 lists: {e}")

        plt.title(title)
        plt.savefig(os.path.join(output_dir, f"{prefix}.png"))
        plt.close()

    plot_and_save_venn(pred_sets, labels, f"Predicted Positive Overlap ({dataset_name})", f"{dataset_name}_venn_predicted")
    plot_and_save_venn(correct_sets, labels, f"Correct Predictions Overlap ({dataset_name})", f"{dataset_name}_venn_correct")
    plot_and_save_venn(error_sets, labels, f"Error Predictions Overlap ({dataset_name})", f"{dataset_name}_venn_error")

def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Loading protein sequences from {args.dataset}...")
    seq_dict = load_sequences(args.dataset)

    results = []
    model_labels = []

    for folder in args.analyze_folder:
        print(f"Loading {folder}...")
        data = load_data(folder)
        if data:
            results.append(data)
            # Use folder name as label
            model_labels.append(os.path.basename(os.path.normpath(folder)))

    if not results:
        print("No valid data loaded.")
        return

    datasets = ['c2', 'c3']
    
    for ds in datasets:
        print(f"Processing dataset: {ds}")
        dfs = []
        valid_labels = []
        
        for data, label in zip(results, model_labels):
            df = get_predictions_df(data, ds)
            if df is not None:
                dfs.append(df)
                valid_labels.append(label)
        
        if not dfs:
            print(f"No data for dataset {ds}")
            continue

        # Align dataframes on pair_id to ensure fair comparison
        # Find common pair_ids
        common_ids = set(dfs[0]['pair_id'])
        for df in dfs[1:]:
            common_ids &= set(df['pair_id'])
        
        print(f"Common samples in {ds}: {len(common_ids)}")
        
        # Filter DFs to common IDs
        filtered_dfs = []
        for df in dfs:
            # Reorder them to match common_ids list order for plotting consistency?
            # Actually, keeping them as DataFrames with filtered rows is enough.
            # But the order in plot lines doesn't matter much as long as pairs match.
            # But for save_detailed_comparison we use set_index, so generic filter is fine.
            filtered_dfs.append(df[df['pair_id'].isin(common_ids)].reset_index(drop=True))
        
        dfs = filtered_dfs

        # Plots
        plot_sorted_probabilities(dfs, valid_labels, args.output_dir, ds)
        plot_logit_distribution(dfs, valid_labels, args.output_dir, ds)
        plot_probability_distribution(dfs, valid_labels, args.output_dir, ds)
        plot_aupr(dfs, valid_labels, args.output_dir, ds)
        
        analyze_top_k_metrics(dfs, valid_labels, args.output_dir, ds)
        analyze_top_percentile_metrics(dfs, valid_labels, args.output_dir, ds)

        perform_venn_analysis(dfs, valid_labels, args.output_dir, ds, seq_dict)

if __name__ == "__main__":
    main()
