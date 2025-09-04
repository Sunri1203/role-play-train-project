import json
import random

# 讀取整個檔案
with open('SFT_senpai_messages_daily_200.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 隨機打亂資料
random.shuffle(data)

# 訓練集與驗證集的分割
# 取前 50 筆作為驗證集
validation_set = data[:50]
# 剩下的 150 筆作為訓練集
training_set = data[50:]

# 將驗證集儲存到新檔案中
with open('SFT_senpai_validation_50.jsonl', 'w', encoding='utf-8') as f:
    for item in validation_set:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 將訓練集儲存到新檔案中
with open('SFT_senpai_training_150.jsonl', 'w', encoding='utf-8') as f:
    for item in training_set:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("驗證集已儲存到 SFT_senpai_validation_50.jsonl")
print("訓練集已儲存到 SFT_senpai_training_150.jsonl")