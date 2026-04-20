import os

# 读取失败的蛋白质ID列表
with open('/home/remilia/BFSW/SEPAL-PPI/mutifeature/Human/failed_prostt_proteins.txt', 'r') as f:
    protein_ids = [line.strip() for line in f.readlines()]

# PDB文件目录
pdb_dir = '../../dataset/Human/2193/pdb'

# 标准氨基酸的三个字母代码
standard_aa = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

# 常见的DNA/RNA残基
nucleotides = {
    'DA', 'DC', 'DG', 'DT', 'DU',  # DNA
    'A', 'C', 'G', 'U', 'I',       # RNA
    'ADE', 'CYT', 'GUA', 'THY', 'URA'  # 其他表示方式
}

def find_non_standard_residues_in_pdb(pdb_file):
    """查找PDB文件中的非标准残基"""
    all_residues = set()
    standard_count = 0
    non_standard_count = 0
    
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                # 关注ATOM记录
                if line.startswith('ATOM'):
                    residue_name = line[17:20].strip()
                    all_residues.add(residue_name)
                    # 分类残基类型
                    if residue_name in standard_aa:
                        standard_count += 1
                    elif residue_name in nucleotides:
                        non_standard_count += 1
                    elif residue_name.isalpha() and len(residue_name) <= 3:
                        # 其他可能的非标准残基
                        non_standard_count += 1
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        
    # 找出非标准残基
    non_standard_residues = all_residues - standard_aa
    # 过滤掉可能是水分子或离子的残基
    non_standard_residues = {res for res in non_standard_residues if res not in {'HOH', 'H2O', 'WAT'}}
    
    return non_standard_residues, standard_count, non_standard_count, all_residues

# 统计结果
total_proteins_with_non_standard = 0
proteins_with_non_standard = []

print("检查蛋白质中的非标准残基...")
print("=" * 60)

for protein_id in protein_ids[:10]:  # 先检查前10个蛋白质
    pdb_file = os.path.join(pdb_dir, f"{protein_id}.pdb")
    if os.path.exists(pdb_file):
        non_standard_residues, std_count, non_std_count, all_residues = find_non_standard_residues_in_pdb(pdb_file)
        print(f"{protein_id}:")
        print(f"  总残基数: {len(all_residues)}, 标准氨基酸: {std_count}, 非标准残基: {non_std_count}")
        if non_standard_residues:
            total_proteins_with_non_standard += 1
            proteins_with_non_standard.append((protein_id, non_standard_residues))
            print(f"  非标准残基: {sorted(non_standard_residues)}")
        else:
            print(f"  非标准残基: 无")
        print()
    else:
        print(f"PDB file not found for {protein_id}")

print("=" * 60)
print(f"在前10个检查的蛋白质中，有 {total_proteins_with_non_standard} 个含有非标准残基")
print()

# 如果用户想检查所有蛋白质，可以取消下面这段代码的注释
"""
print("\n检查所有蛋白质中的非标准残基...")
print("=" * 60)

for protein_id in protein_ids:
    pdb_file = os.path.join(pdb_dir, f"{protein_id}.pdb")
    if os.path.exists(pdb_file):
        non_standard_residues, std_count, non_std_count, all_residues = find_non_standard_residues_in_pdb(pdb_file)
        if non_standard_residues:
            total_proteins_with_non_standard += 1
            proteins_with_non_standard.append((protein_id, non_standard_residues))

print(f"在全部70个蛋白质中，有 {total_proteins_with_non_standard} 个含有非标准残基")

if proteins_with_non_standard:
    print("\n含有非标准残基的蛋白质详情:")
    for protein_id, non_standard_residues in proteins_with_non_standard:
        print(f"  {protein_id}: {', '.join(sorted(non_standard_residues))}")
else:
    print("没有发现含有非标准残基的蛋白质")
"""