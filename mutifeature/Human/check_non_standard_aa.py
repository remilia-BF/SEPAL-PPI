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

def find_non_standard_aa_in_pdb(pdb_file):
    """查找PDB文件中的非标准氨基酸"""
    non_standard_aa = set()
    
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                # 关注ATOM记录
                if line.startswith('ATOM'):
                    residue_name = line[17:20].strip()
                    # 检查是否为非标准氨基酸
                    if residue_name not in standard_aa and residue_name.isalpha() and len(residue_name) == 3:
                        # 只考虑长度为3的字母名称，避免数字或其他符号
                        non_standard_aa.add(residue_name)
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        
    return non_standard_aa

# 统计结果
total_proteins_with_non_standard = 0
proteins_with_non_standard = []

print("检查蛋白质中的非标准氨基酸...")
print("=" * 50)

for protein_id in protein_ids:
    pdb_file = os.path.join(pdb_dir, f"{protein_id}.pdb")
    if os.path.exists(pdb_file):
        non_standard_aa = find_non_standard_aa_in_pdb(pdb_file)
        if non_standard_aa:
            total_proteins_with_non_standard += 1
            proteins_with_non_standard.append((protein_id, non_standard_aa))
            print(f"{protein_id}: {non_standard_aa}")
    else:
        print(f"PDB file not found for {protein_id}")

print("=" * 50)
print(f"总计: {total_proteins_with_non_standard} 个蛋白质含有非标准氨基酸")
print()

# 详细列出含有非标准氨基酸的蛋白质
if proteins_with_non_standard:
    print("含有非标准氨基酸的蛋白质详情:")
    for protein_id, non_standard_aa in proteins_with_non_standard:
        print(f"  {protein_id}: {', '.join(non_standard_aa)}")
else:
    print("没有发现含有非标准氨基酸的蛋白质")