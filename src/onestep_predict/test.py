import torch
from esme import ESM2
from esme.alphabet import tokenize

# 1. 加载模型
device = 0 if torch.cuda.is_available() else "cpu"
# 请确保路径指向你的 .safetensors 文件
model = ESM2.from_pretrained("cache/esm/esm2_15b.safetensors", device=device)
model.eval()

# 2. 准备数据
sequences = ['MEEPQSDPSVEPPLSQESTFSLDLWK', 'MADQLTEEQIAEFKEAFSLFDKDG']
tokens = tokenize(sequences).to(device)

print(f"输入 Tokens 形状: {tokens.shape}")

# --- 方法一：观察默认输出 ---
with torch.no_grad():
    output = model(tokens)
    # 在 esme 库中，默认返回的是 logits [B, L, Alphabet_Size]
    # 如果形状的最后一维等于模型隐藏维度（如 320, 480, 1280），那它就是 Embedding
    print(f"默认输出形状: {output.shape}")

# --- 方法二：使用 Hook 强制捕获 Transformer 最后一层的输出 ---
# 这是最稳妥的方法，不需要知道 API 参数名也能拿到结果
activations = {}

def get_activation(name):
    def hook(model, input, output):
        # 如果输出是 tuple，取第一个元素（通常是 tensor）
        if isinstance(output, tuple):
            activations[name] = output[0].detach()
        else:
            activations[name] = output.detach()
    return hook

# 在 esme 中，最后一层通常是 model.layers[-1] 或 model.transformer.layers[-1]
# 我们挂载到最后一个 transformer 块的输出上
last_layer_module = model.layers[-1] 
handle = last_layer_module.register_forward_hook(get_activation('last_layer_emb'))

# 执行前向传播
with torch.no_grad():
    _ = model(tokens)

# 移除 Hook
handle.remove()

# 3. 结果观察
last_emb = activations['last_layer_emb']
print("-" * 30)
print(f"最后一层 Embedding 形状: {last_emb.shape}") 
# 预期形状: [Batch, Seq_Len, Hidden_Dim]
print(f"数值样本 (前三个维度): \n{last_emb[0, 0, :3]}")