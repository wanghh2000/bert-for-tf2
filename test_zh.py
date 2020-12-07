import os
import bert

model_dir = r'C:/bd_ai/dli/models/bert/chinese_L-12_H-768_A-12'

# 分词
vocab_file = os.path.join(model_dir, "vocab.txt")
tokenizer = bert.albert_tokenization.FullTokenizer(vocab_file=vocab_file)
#tokens = tokenizer.tokenize(u"你好世界")
tokens = tokenizer.tokenize(u"IBM正利用人工智能技术、云计算、区块链、物联网、助力各行业重铸商业模式。立即登录IBM网站了解更多成功案例。")
print(tokens)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)