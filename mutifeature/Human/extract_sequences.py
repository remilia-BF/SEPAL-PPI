import os

# 读取失败的蛋白质ID列表
with open('/home/remilia/BFSW/SEPAL-PPI/mutifeature/Human/failed_prostt_proteins.txt', 'r') as f:
    protein_ids = [line.strip() for line in f.readlines()]

# PDB文件目录
pdb_dir = '../../dataset/Human/2193/pdb'

# 三个字母代码到一个字母代码的氨基酸映射
aa_map = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # 处理可能的非标准氨基酸
    'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z'
}

def extract_sequence_from_pdb(pdb_file):
    """从PDB文件中提取氨基酸序列"""
    sequence = []
    last_residue_number = None
    
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                # 只关注ATOM记录且是CA原子（每个氨基酸只有一个CA原子）
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    residue_number = int(line[22:26])
                    # 避免重复计数同一个残基（有些PDB文件中同一残基有多个记录）
                    if residue_number != last_residue_number:
                        residue_name = line[17:20].strip()
                        if residue_name in aa_map:
                            sequence.append(aa_map[residue_name])
                        else:
                            # 如果遇到未识别的氨基酸，用'X'表示
                            sequence.append('X')
                        last_residue_number = residue_number
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return None
        
    return ''.join(sequence)

# 生成FASTA文件
fasta_file = '/home/remilia/BFSW/SEPAL-PPI/mutifeature/Human/failed_proteins_sequences.fasta'
with open(fasta_file, 'w') as f_out:
    for protein_id in protein_ids:
        pdb_file = os.path.join(pdb_dir, f"{protein_id}.pdb")
        if os.path.exists(pdb_file):
            sequence = extract_sequence_from_pdb(pdb_file)
            if sequence:
                f_out.write(f">{protein_id}\n")
                # 每60个字符换行，符合FASTA格式规范
                for i in range(0, len(sequence), 60):
                    f_out.write(sequence[i:i+60] + '\n')
            else:
                print(f"Could not extract sequence for {protein_id}")
        else:
            print(f"PDB file not found for {protein_id}")

print(f"FASTA file generated: {fasta_file}")