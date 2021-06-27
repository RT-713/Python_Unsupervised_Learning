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
# 相関行列の作成
import numpy as np
from scipy.stats import pearsonr

# 空のdfを用意
correlationMatrix = pd.DataFrame(data=[], index=dataX.columns, columns=dataX.columns)

# dfに相関係数を計算した結果を順次代入
for i in dataX.columns:
    for j in dataX.columns:
        correlationMatrix.loc[i, j] = np.round(pearsonr(dataX.loc[:, i], dataX.loc[:, j])[0], 2)
correlationMatrix
# %% [markdown]
# ## データの可視化
# - アンバランスなデータを確認する
# %%
# Class列（Series）の一意の値を数える
count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
count_classes
# %%
# 文字化け対応（フォント指定）
sns.set(font='Hiragino Sans')

# x軸は0, 1として，y軸は比率とする．
ax = sns.barplot(x=count_classes.index, y=count_classes/len(data))

# グラフのタイトル・x軸，y軸の名前
ax.set_title('クラスの出現頻度割合')
ax.set_xlabel('Class')
ax.set_ylabel('頻度の割合')
plt.show()
# %%
