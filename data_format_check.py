from transformers import DataCollatorForLanguageModeling, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, mlm_probability=0.25)

# 为了重现性，设置随机种子
import torch
torch.manual_seed(42)

# 两个句子
texts = ["I love watching movies", "She enjoys reading books"]
tokenized = tokenizer(texts, padding=False, truncation=True)  # 不填充

print("原始分词结果:")
for i in range(len(texts)):
    print(f"句子{i+1} tokens: {tokenized['input_ids'][i]}")
    print(f"句子{i+1} 文本: {tokenizer.decode(tokenized['input_ids'][i])}")
    print()

# 转换为DataCollator需要的格式
batch = [{"input_ids": ids} for ids in tokenized["input_ids"]]

# 处理批次
processed = data_collator(batch)

print("处理后结果:")
for i in range(len(batch)):
    input_ids = processed["input_ids"][i]
    labels = processed["labels"][i]
    
    print(f"\n句子{i+1}:")
    print(f"input_ids: {input_ids.tolist()}")
    print(f"labels:    {labels.tolist()}")
    
    # 找出需要预测的位置
    for j, (inp, lbl) in enumerate(zip(input_ids, labels)):
        if lbl != -100:
            masked_word = tokenizer.decode([inp]) if inp == tokenizer.mask_token_id else tokenizer.decode([inp])
            true_word = tokenizer.decode([lbl])
            print(f"  → 位置{j}需要预测: '{masked_word}' → '{true_word}'")