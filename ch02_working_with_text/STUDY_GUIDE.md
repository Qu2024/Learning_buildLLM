# 第2章：Working with Text Data - 学习指导

## 本章目标

将**原始文本**转化为**可供 LLM 训练使用的嵌入向量**，构建完整的数据处理 pipeline：

```
原始文本 → 分词 → Token IDs → 嵌入向量 + 位置编码 → 模型输入
```

---

## 学习路线（共 7 个小节，建议 2-3 天完成）

### 📖 2.1 理解词嵌入（Word Embeddings）

**核心概念：**
- 为什么需要嵌入？— 神经网络只能处理数字，不能直接处理文本
- 嵌入的本质：把离散的 token 映射到连续的高维向量空间
- Word2Vec vs LLM 嵌入的区别：LLM 的嵌入是训练过程中学习得到的，不是预先固定的

**思考题：**
- 为什么不用 one-hot 编码代替嵌入？（提示：维度灾难 + 无法表达语义相似性）

---

### 📖 2.2 文本分词（Tokenizing Text）

**动手任务：** 写一个基于正则表达式的简单分词器

**关键步骤：**
1. 下载样本文本 `the-verdict.txt`（Edith Wharton 的短篇小说）
2. 用正则表达式分割文本，处理标点和空格
3. 理解分词的关键决策：标点是否独立为 token？空格怎么处理？

**核心正则：**
```python
import re
text = "Hello, world. Is this-- a test?"
tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
tokens = [t.strip() for t in tokens if t.strip()]
# ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

**练习：**
- 尝试不同的正则模式，观察分词结果的差异
- 思考：为什么 `--` 需要被单独处理？

---

### 📖 2.3 Token 转 Token ID

**动手任务：** 构建词汇表（vocabulary），实现 token ↔ ID 双向映射

**关键步骤：**
1. 收集所有唯一 token 并排序
2. 构建 `token → id` 和 `id → token` 的映射字典
3. 实现 `SimpleTokenizerV1` 类（encode/decode 方法）

**代码骨架：**
```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [t.strip() for t in tokens if t.strip()]
        return [self.str_to_int[t] for t in tokens]

    def decode(self, ids):
        tokens = [self.int_to_str[i] for i in ids]
        return " ".join(tokens)
```

**练习：**
- 对一段文本做 encode → decode 的往返测试，看能否还原
- 故意输入一个词汇表中不存在的词，观察报错

---

### 📖 2.4 添加特殊 Token

**动手任务：** 实现 `SimpleTokenizerV2`，支持特殊 token

**关键概念：**
- `<|unk|>` — 处理未知词（out-of-vocabulary）
- `<|endoftext|>` — 标记文本段落的边界（GPT 训练时连接不同文档用）

**思考题：**
- GPT 为什么需要 `<|endoftext|>`？（提示：预训练时多篇文档拼接在一起）
- BERT 的 `[CLS]`/`[SEP]` 和 GPT 的特殊 token 有什么区别？

---

### 📖 2.5 字节对编码（Byte Pair Encoding, BPE）⭐ 重要

**核心概念：**
- BPE 是 GPT 系列实际使用的分词算法
- 核心思想：从字符级别开始，反复合并最频繁出现的字符对
- 优点：很好地平衡了词汇量大小和 OOV（未知词）问题

**动手任务：** 使用 `tiktoken` 库体验 GPT 的 BPE 分词

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, do you like tea? <|endoftext|> In the sunlit"

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250]

print(tokenizer.decode(integers))
# "Hello, do you like tea? <|endoftext|> In the sunlit"
```

**练习：**
- 对比 SimpleTokenizerV2 和 tiktoken 对同一文本的分词结果
- 输入一个罕见的英文单词（如 "Akwirw ier"），看 BPE 如何拆分为子词
- 尝试中文文本，观察 BPE 是怎么处理的

**深入理解 BPE 算法：**
```
初始：["l", "o", "w", " ", "l", "o", "w", "e", "r"]
第1轮：最频繁的对是 ("l","o") → 合并为 "lo"
第2轮：最频繁的对是 ("lo","w") → 合并为 "low"
...以此类推
```

---

### 📖 2.6 滑动窗口采样（Data Sampling with Sliding Window）⭐⭐ 核心

**核心概念：**
- LLM 训练的本质是 **next-token prediction**（预测下一个 token）
- 用滑动窗口生成训练样本：input → target（target 就是 input 右移一位）

**图解：**
```
文本: [The, cat, sat, on, the, mat]
context_size = 4

样本1: input=[The, cat, sat, on]   → target=[cat, sat, on, the]
样本2: input=[cat, sat, on, the]   → target=[sat, on, the, mat]
```

**动手任务：** 实现 `GPTDatasetV1` 和 `create_dataloader_v1`

```python
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

**练习：**
- `max_length` 和 `stride` 分别控制什么？stride < max_length 时会怎样？
- 把 stride 设为 1 和设为 max_length，分别打印前 5 个样本，对比差异
- 思考：stride 小意味着更多重叠样本，这对训练有什么影响？

---

### 📖 2.7 Token 嵌入 + 位置编码 ⭐⭐ 核心

**核心概念：**
- **Token Embedding**：把 token ID 映射为向量（可学习参数）
- **Positional Embedding**：给每个位置编码，让模型知道 token 的顺序
- **最终输入 = Token Embedding + Positional Embedding**

**动手任务：**
```python
# 假设词汇量 50257，嵌入维度 256，上下文长度 1024
vocab_size = 50257
output_dim = 256
context_length = 1024

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# 输入 token IDs: batch_size=1, seq_len=4
token_ids = torch.tensor([[2, 3, 5, 1]])

# Token 嵌入
token_embeddings = token_embedding_layer(token_ids)  # (1, 4, 256)

# 位置嵌入
positions = torch.arange(4)  # [0, 1, 2, 3]
pos_embeddings = pos_embedding_layer(positions)       # (4, 256)

# 最终输入
input_embeddings = token_embeddings + pos_embeddings  # (1, 4, 256) 广播相加
```

**练习：**
- 打印 token_embeddings 和 pos_embeddings 的 shape，确认你理解每个维度的含义
- 思考：为什么位置编码是**可学习**的而不是固定的正弦函数？（GPT vs 原始 Transformer）
- 修改 output_dim 为不同值（64, 128, 768），观察参数量的变化

---

## 🧪 综合练习

完成所有小节后，尝试把整个 pipeline 串起来：

```python
# 完整 pipeline
raw_text = open("data/the-verdict.txt").read()

# 1. 分词
tokenizer = tiktoken.get_encoding("gpt2")

# 2. 创建 DataLoader
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4,
    stride=4, shuffle=False
)

# 3. 取一个 batch
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(f"Inputs shape:  {inputs.shape}")   # (8, 4)
print(f"Targets shape: {targets.shape}")  # (8, 4)

# 4. 嵌入
token_emb = token_embedding_layer(inputs)        # (8, 4, 256)
pos_emb = pos_embedding_layer(torch.arange(4))   # (4, 256)
input_embeddings = token_emb + pos_emb            # (8, 4, 256)

print(f"Final input shape: {input_embeddings.shape}")
# 这就是送入 Transformer 的输入！
```

---

## 📝 本章核心 Checklist

学完后确认你能回答以下问题：

- [ ] 为什么需要 tokenization？为什么不直接用字符级别？
- [ ] BPE 算法的核心思想是什么？它如何处理未见过的词？
- [ ] `<|endoftext|>` token 的作用是什么？
- [ ] 滑动窗口的 `max_length` 和 `stride` 各自控制什么？
- [ ] Token Embedding 和 Positional Embedding 为什么要相加而不是拼接？
- [ ] `nn.Embedding(50257, 768)` 有多少可学习参数？（答案：50257 × 768 ≈ 3860 万）
- [ ] 输入 shape 为 (batch_size, seq_len) 时，经过嵌入后输出 shape 是什么？

---

## 📦 参考资源

- [官方代码 ch02.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb)
- [tiktoken 文档](https://github.com/openai/tiktoken)
- [BPE 算法详解（原论文）](https://arxiv.org/abs/1508.07909)
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) — 理解嵌入的好文章
