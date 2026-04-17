import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
import time

# ==========================
# 特征工程
# ==========================

def create_features(df):
    """创建所有特征（无数据泄露版本）"""
    df = df.copy()
    
    # 基础收益率
    df['return_1h'] = df['close'].pct_change(1)
    df['return_4h'] = df['close'].pct_change(4)
    df['return_24h'] = df['close'].pct_change(24)
    
    # 成交量特征
    df['volume_change'] = df['volume'].pct_change(1)
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
    
    # 价格位置（使用shift避免泄露）
    high_low_range = df['high'] - df['low']
    high_low_range = high_low_range.replace(0, np.nan)
    df['h1_position'] = ((df['close'] - df['low']) / high_low_range).shift(1)
    
    # 均线系统（全部shift）
    df['ma7'] = df['close'].rolling(7).mean().shift(1)
    df['ma24'] = df['close'].rolling(24).mean().shift(1)
    df['price_vs_ma7'] = (df['close'].shift(1) - df['ma7']) / df['ma7']
    df['price_vs_ma24'] = (df['close'].shift(1) - df['ma24']) / df['ma24']
    df['ma7_slope'] = df['ma7'].pct_change(1).shift(1)
    
    # RSI（使用shift避免泄露）
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean().shift(1)
    loss = (-delta.clip(upper=0)).rolling(14).mean().shift(1)
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 布林带（全部shift）
    rolling_std = df['close'].rolling(20).std().shift(1)
    bb_upper = df['ma24'] + 2 * rolling_std
    bb_lower = df['ma24'] - 2 * rolling_std
    df['bb_position'] = (df['close'].shift(1) - bb_lower) / (bb_upper - bb_lower)
    
    # 波动率
    df['volatility'] = df['return_1h'].rolling(24).std().shift(1)
    df['atr'] = (df['high'] - df['low']).rolling(14).mean().shift(1) / df['close'].shift(1)

    # 价格加速度（动量的变化率）
    df['return_acceleration'] = df['return_1h'].diff(1)

    # 成交量与价格的背离
    df['volume_price_divergence'] = df['volume_change'] * df['return_1h']
    df['divergence_signal'] = ((df['volume_change'] > 0.1) & (df['return_1h'] < -0.01)).astype(int)

    # 连续上涨/下跌计数
    df['consecutive_up'] = (df['return_1h'] > 0).astype(int).groupby(
        (df['return_1h'] > 0).astype(int).diff().ne(0).cumsum()
    ).cumsum()
    df['consecutive_down'] = (df['return_1h'] < 0).astype(int).groupby(
        (df['return_1h'] < 0).astype(int).diff().ne(0).cumsum()
    ).cumsum()

    # 市场宽度（如果有多币种数据）
    # 这里简化：用成交量加权价格
    df['vwap_deviation'] = (df['close'] - df['volume'].rolling(24).apply(
        lambda x: np.average(df.loc[x.index, 'close'], weights=x)
    )) / df['close']

    # 波动率比率
    df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(24).mean()
    
    # 时间特征
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek


    
    # 删除NaN
    df = df.dropna()
    
    return df

# ==========================
# 目标变量
# ==========================

def create_target(df, threshold=0.002):
    """创建目标变量，不再过滤数据"""
    future_return = (df['close'].shift(-1) - df['close']) / df['close']
    
    # 简单二分类（涨/跌）
    df['target'] = (future_return > 0).astype(int)
    
    # 可选：三分类（强烈涨/横盘/强烈跌）
    # df['target'] = np.where(future_return > threshold, 2,
    #                         np.where(future_return < -threshold, 0, 1))
    
    return df.dropna()





# ==========================
# 1. PULL DATA
# ==========================

exchange = ccxt.okx()

# ohlcv -> Open, High, Low, Close, Volume
# K -> 1h, Limit 500 K
'''
ohlcv = exchange.fetch_ohlcv(
    symbol = 'BTC/USDT',
    timeframe = '1h',
    limit = 500         # For 15min, 500 -> 2000
)
'''

def fetch_ohlcv_full(symbol, timeframe, total_limit):
    ''' 分批拉取k线数据 '''
    all_ohlcv = []
    # since = None # from the very begining

    # first 300 lines
    batch = exchange.fetch_ohlcv(
        symbol = symbol,
        timeframe = timeframe,
        limit = 300
    )

    all_ohlcv = batch
    print(f"Got {len(all_ohlcv)} K Lines ...")

    # PAGE TURN
    while len(all_ohlcv) < total_limit:

        # existing earliest data as timesatmp
        earliest_time = all_ohlcv[0][0]

        # calculate 300 line timestamp
        # 1h = 3600s = 3600000ms
        ''' CHANGE HOUR '''
        timeframe_ms = 3600 * 1000
        since = earliest_time - 300 * timeframe_ms

        # 300 each time
        batch = exchange.fetch_ohlcv(
            symbol = symbol,
            timeframe = timeframe,
            limit = 300,
            since = since,
            # params = {'before': earliest_time} # OKX api, get data before this timestamp
        )

        if not batch:
            break  # no further data

        # keep earlier data, in case duplicate
        batch = [x for x in batch if x[0] < earliest_time]

        if not batch:
            break  # no further data

        all_ohlcv = batch + all_ohlcv
        print(f"Got {len(all_ohlcv)} K Lines ...")
        time.sleep(1)


    return all_ohlcv[-total_limit:]   # Keep new N lines 

# Call method ohlcv
''' CHANGE HOUR '''
ohlcv = fetch_ohlcv_full('BTC/USDT', '1h', 2000)

# get raw data
df = pd.DataFrame(ohlcv, columns = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume'
])

# turn into a readable table
# delete timestamp and sort base on ms time
df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.drop('timestamp', axis = 1)
df = df.set_index('time')

print(f"Get {len(df)} K-Line")
print(df.tail(3))



# ==========================
# Clean
# ==========================

df = df.sort_index()

print(f"{len(df)} Lines before clean")

# 1. 删除重复的时间戳
df = df[~df.index.duplicated(keep='first')]

# 2. 删除成交量为0的K
df = df[df['volume'] > 0]

# 3. 删除价格为0或负数的行
df = df[df['close'] > 0]
df = df[df['high'] > 0]
df = df[df['low'] > 0]

# 4. 删除高地价逻辑错误的行
df = df[df['high'] >= df['low']]

# Build Specs
df = create_features(df)

# Target Vars
df = create_target(df)


print(f"\n特征构造完成, 共 {len(df)} 条有效数据")
print(f"上涨比例: {df['target'].mean():.1%}")



# ==========================
# 4. Train XGBoost
# ==========================

features = [
    'return_1h', 
    'return_4h', 
    'return_24h', 
    'volume_change', 
    'volume_ma_ratio',
    'h1_position', 
    # 'upper_shadow', -> h1_position
    # price-scale dependent problem
    # 'ma7',
    # 'ma24',
    'price_vs_ma7',
    'price_vs_ma24',
    'ma7_slope', 
    'rsi',
    'bb_position',
    'volatility',
    'atr',
    'return_acceleration',
    'volume_price_divergence',
    'divergence_signal',
    'consecutive_up',
    'consecutive_down',
    'vwap_deviation',
    'volatility_ratio',
    'hour',
    'day_of_week'

]

X = df[features] # features - > model input
y = df['target'] # tags -> predict anwsers

# 按时间顺序切分，不能随机切分（时间序列数据的规则）
# ├── 前80% → X_train / y_train  用来训练
# └── 后20% → X_test  / y_test   用来测试
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 过拟合 -> 把训练数据的噪音也记住
# 核心原则只有一个： 训练集要够大让模型学到规律，测试集要够大让评估结果可信。
model = xgb.XGBClassifier(
    n_estimators = 200,       # 种100棵树,太大会过拟合 # 100
    max_depth = 3,          # 每棵树最多3层，防止过拟合 # 3
    learning_rate = 0.05,    # 学习速度，越小越稳 (每棵树的贡献权重) # 0.1
    subsample = 0.8,
    colsample_bytree = 0.8,
    reg_alpha = 0.3,
    reg_lambda = 0.5,
    min_child_weight = 3,
    scale_pos_weight = sum(y_train == 0)/sum(y_train == 1),
    random_state = 42
    # n_jobs = 1
)

model.fit(X_train, y_train)



# ==========================
# 5. Assess the Model
# ==========================

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.1%}")

cm = confusion_matrix(y_test, y_pred)
print(cm)

# 特征重要性
importance_df = pd.DataFrame({
    'Features': features,
    'Importance': model.feature_importances_ 
}).sort_values('Importance', ascending = False)
# feature_importances_ 是模型训练完之后自动计算的，告诉每个特征贡献了多少。

# sort_values('重要性', ascending=False) 从高到低排序：
print("\nFeature Importances: ")
print(importance_df.to_string(index = False))



# ==========================
# 6. Predict current newest K
# ==========================

latest = X.iloc[[-1]]   # Newest
prob = model.predict_proba(latest)[0]

print(f"\n Current Prediction on BTC for NEXT 1 hour: ")
print(f"    Increase P: {prob[1]:.1%}")
print(f"    Decrease P: {prob[0]:.1%}")

if prob[1] > 0.6:
    print("    Signal: UP")
elif prob[0] > 0.6:
    print("    Signal: DOWN")
else:
    print("    Signal: UNCLEAR")
