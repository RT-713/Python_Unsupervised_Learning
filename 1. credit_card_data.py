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
# %%
# %%
