# %% [markdown]
# ## 機械学習システム Part1
# %%
# 最初のみ実行
# # データの読み込み
# data_row = pd.read_csv('./data/credit_card_data/creditcard.csv')
# # データの変換（csv → pkl）
# data_row.to_pickle('./data/credit_card_data/creditcard.pkl')
# %%
import pandas as pd

# pklデータでの読み込み
data = pd.read_pickle('./data/CreditCard/creditcard.pkl')
data.head()
# %%
# 要約統計量
data.describe()
# %%
# データ列の確認
data.columns
# %%
# データの欠損値確認
data.isnull().sum()
# %%
# 固有の値（ユニークな値）の確認
distinctCounter = data.apply(lambda x: len(x.unique()))
distinctCounter
# %% [markdown]
# ## 不正なトランザクションを確認
# 不正なトランザクション＝「Class列＝１」
# %%
print(f'不正なトランザクションの数：', data['Class'].sum())
# %% [markdown]
# ## 特徴量行列とラベル配列の作成
# %%
# dataXにはラベルを除いたdfを，dataYにはClassの配列データを格納
dataX = data.copy().drop(['Class'], axis=1)
dataY = data['Class'].copy()
# %%
# 特徴量の標準化処理
from sklearn.preprocessing import StandardScaler

# Time列のデータは標準化の対象外として処理
featuresToScale = dataX.drop(['Time'], axis=1).columns
sX = StandardScaler(copy=True)
dataX.loc[:, featuresToScale] = sX.fit_transform(dataX[featuresToScale])

# Time列を除く列が平均0, 標準偏差1になっている（＝標準化されている）ことを確認
dataX.describe()
# %%
# 相関をヒートマップで確認
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
# annot：数値の表示，square：正方形で表示
sns.heatmap(dataX.corr(), annot=False, square=True)
# %%
