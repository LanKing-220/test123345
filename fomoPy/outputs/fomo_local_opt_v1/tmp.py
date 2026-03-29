import json
import matplotlib.pyplot as plt

# 读取训练历史
with open('fomoPy/outputs/fomo_local_opt_v1/training_history.json', 'r', encoding='utf-8') as f:
    history = json.load(f)

lr = history['lr']

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(lr)+1), lr, marker='o')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.xticks(range(1, len(lr)+1))
plt.tight_layout()
plt.show()
