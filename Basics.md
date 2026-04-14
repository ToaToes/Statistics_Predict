1. Python and ENV

```
pip install xgboost pandas numpy scikit-learn matplotlib ccxt jupyter
```

xgboost — 模型本体
pandas / numpy — 数据处理
scikit-learn — 配套工具（数据分割、评估指标）
matplotlib — 画图
ccxt — 连接OKX等交易所API
jupyter — 方便你边跑边看结果


2. 

特征重要性（Feature Importance）

先理解"偏重"是什么意思
XGBoost 训练完之后，会自动计算：

每个特征对最终预测结果贡献了多少？

比如训练完BTC预测模型后，可能发现：
```
成交量变化    ████████████  42%  ← 最重要
资金费率      ████████      28%
过去1h涨跌    █████         18%
过去4h涨跌    ███           12%
```
说明这个模型主要靠成交量来判断涨跌，而不是价格本身。

反映到代码上
```
pythonimport xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 假设这是你的特征和标签
# 真实项目里这些数据来自OKX API
data = {
    'volume_change':  [0.3, -0.1, 0.5, -0.3, 0.2],   # 成交量变化
    'price_1h':       [0.02, -0.01, 0.005, -0.03, 0.01],  # 过去1h涨跌
    'price_4h':       [0.01, -0.02, 0.03, -0.01, 0.005],  # 过去4h涨跌
    'funding_rate':   [0.01, 0.02, 0.005, 0.05, 0.01],    # 资金费率
}
y = [1, 0, 1, 0, 1]  # 下一小时涨(1)跌(0)

X = pd.DataFrame(data)

# 训练模型
model = xgb.XGBClassifier(n_estimators=100)
model.fit(X, y)

# ====== 查看偏重 ======
importance = model.feature_importances_
# 输出类似：[0.42, 0.18, 0.12, 0.28]
# 对应顺序：成交量  1h涨跌  4h涨跌  资金费率

# 用表格看更清楚
importance_df = pd.DataFrame({
    '特征': X.columns,
    '重要性': importance
}).sort_values('重要性', ascending=False)

print(importance_df)
```

输出：
        特征       重要性 <br>
0  volume_change   0.42 <br>
3   funding_rate   0.28 <br>
1       price_1h   0.18 <br>
2       price_4h   0.12 <br>


画出来更直观
```
pythonplt.figure(figsize=(8, 4))
plt.barh(importance_df['特征'], importance_df['重要性'])
plt.xlabel('重要性')
plt.title('每个特征对预测的贡献')
plt.tight_layout()
plt.show()
```
会画出这样的图：
```
volume_change  ████████████  0.42
 funding_rate  ████████      0.28
     price_1h  █████         0.18
     price_4h  ███           0.12
```

这个信息能用来做什么 

1. 删掉没用的特征 <br>
python# 重要性低于5%的特征直接删掉 <br>

# 减少噪音，模型更准
```
useful_features = importance_df[importance_df['重要性'] > 0.05]['特征'].tolist()
X_clean = X[useful_features]
```

2. 指导你去找更多类似特征 <br>
python# 发现成交量最重要

# → 那就多加几个成交量相关特征
# 比如：过去15分钟成交量、过去4小时成交量、异常成交量标记

3. 理解市场规律 <br>
python# 如果资金费率重要性很高

# → 说明期货市场情绪对价格影响大
# → 这本身就是一个有价值的市场洞察

关键思路 <br>
特征重要性高  →  这个指标值得深入研究 <br>
特征重要性低  →  可能是噪音，考虑删掉 <br>
特征重要性=0  →  完全没用，一定删掉 <br>

很多时候，看特征重要性本身比看预测结果更有价值——它告诉你市场到底在乎什么。


3. 

OKX API： 
https://www.okx.com/docs-v5/en/#overview
