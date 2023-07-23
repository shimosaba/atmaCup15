import sys
import numpy as np
import pandas as pd
import random
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

import xgboost
from gensim.models import word2vec
import feature_engineering as fe

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

input_path = '../../input/'
output_path = './seen/'

def add_w2v_features(train_df, val_df, test_df=None, consider_score=True):
    anime_ids = train_df['japanese_name'].unique().tolist()
    user_anime_list_dict = {user_id: anime_ids.tolist() for user_id, anime_ids in train_df.groupby('user_id')['japanese_name']}

    # スコアを考慮する場合
    # 今回は1～10のレーティングなので、スコアが5のアニメは5回、スコアが10のアニメは10回、タイトルをリストに追加する
    if consider_score:
        title_sentence_list = []
        for user_id, user_df in train_df.groupby('user_id'):
            user_title_sentence_list = []
            for anime_id, anime_score in user_df[['japanese_name', 'score']].values:
                for i in range(anime_score):
                    user_title_sentence_list.append(anime_id)
            title_sentence_list.append(user_title_sentence_list)
    # スコアを考慮しない場合
    # タイトルをそのままリストに追加する
    else:
        title_sentence_list = train_df.groupby('user_id')['japanese_name'].apply(list).tolist()

    # ユーザごとにshuffleしたリストを作成
    shuffled_sentence_list = [random.sample(sentence, len(sentence)) for sentence in title_sentence_list]  ## <= 変更点

    # 元のリストとshuffleしたリストを合わせる
    train_sentence_list = title_sentence_list + shuffled_sentence_list

    # word2vecのパラメータ
    vector_size = 64
    random.seed(seed)
    w2v_params = {
        "vector_size": vector_size,  ## <= 変更点
        "epochs": 10,
        "seed": seed,
        "min_count": 1,
        "window": 15, 
        "sg": 1
    }

    # word2vecのモデル学習
    model = word2vec.Word2Vec(train_sentence_list, **w2v_params)

    # ユーザーごとの特徴ベクトルと対応するユーザーID
    user_factors = {user_id: np.mean([model.wv[anime_id] for anime_id in user_anime_list], axis=0) for user_id, user_anime_list in user_anime_list_dict.items()}

    # アイテムごとの特徴ベクトルと対応するアイテムID
    item_factors = {aid: model.wv[aid] for aid in anime_ids}

    # データフレームを作成
    user_factors_df = pd.DataFrame(user_factors).T.reset_index().rename(columns={"index": "user_id"})
    item_factors_df = pd.DataFrame(item_factors).T.reset_index().rename(columns={"index": "japanese_name"})

    # データフレームのカラム名をリネーム
    user_factors_df.columns = ["user_id"] + [f"user_factor_{i}" for i in range(vector_size)]
    item_factors_df.columns = ["japanese_name"] + [f"item_factor_{i}" for i in range(vector_size)]

    train_df = train_df.merge(user_factors_df, on="user_id", how="left")
    train_df = train_df.merge(item_factors_df, on="japanese_name", how="left")

    val_df = val_df.merge(user_factors_df, on="user_id", how="left")
    val_df = val_df.merge(item_factors_df, on="japanese_name", how="left")

    if test_df is not None:
        test_df = test_df.merge(user_factors_df, on="user_id", how="left")
        test_df = test_df.merge(item_factors_df, on="japanese_name", how="left")
        return train_df, val_df, test_df

    return train_df, val_df


def main(seed):
    set_seed(seed)
    anime = pd.read_csv(input_path+'anime.csv')
    train = pd.read_csv(input_path+'train.csv')
    test = pd.read_csv(input_path+'test.csv')
    sub = pd.read_csv(input_path+'sample_submission.csv')
    sub['score'] = 0

    DEBUG = False
    if DEBUG:
        train = train.iloc[:10000]
        test = test.iloc[:10000]

    # filter seen user
    seen_user = train['user_id'].unique()
    tmp_test = test.copy()
    tmp_test = tmp_test[tmp_test['user_id'].isin(seen_user)].reset_index(drop=True)
    tmp_test['score'] = 0

    # feature engineering
    train['class'] = 0
    tmp_test['class'] = 1
    df = pd.concat([train, tmp_test])
    df, ids = fe.seen_add_feature(df, anime, seed)

    ids.to_parquet(output_path+str(seed)+'_ids.parquet')
    df.to_parquet(output_path+str(seed)+'_featured_df.parquet')
    df = pd.read_parquet(output_path+str(seed)+'_featured_df.parquet')
    ids = pd.read_parquet(output_path+str(seed)+'_ids.parquet')

    train = df[df['class']==0]
    test = df[df['class']==1]

    train = train.drop(['class'], axis=1)
    train['score'] = train['score'].astype(int)
    test = test.drop(['score','class'], axis=1)

    FOLD = 5
    skf = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)
    for fold_id, (_, valid_idx) in enumerate(skf.split(train, train["user_id"])):
        train.loc[valid_idx, "fold"] = fold_id


    xgb_params = {
                'n_estimators': 10000,
                'booster': 'gbtree',
                'grow_policy': 'depthwise',
                'n_jobs': -1,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'random_state': seed,
                'tree_method': 'gpu_hist',
                'max_depth': 7, 
                'gamma': 0.3, 
                'subsample': 0.8, 
                'colsample_bytree': 0.8, 
                'reg_alpha': 0.01, 
                'reg_lambda': 0.10, 
                'learning_rate': 0.01
            }

    score_list = []
    valid_dfs = pd.DataFrame()
    for fold in range(FOLD):
        print('='*20 + 'fold : ' +str(fold)+ '='*20)
        train_df = train[train['fold']!=fold].copy()
        valid_df = train[train['fold']==fold].copy()
        test_df = test.copy()

        train_df, valid_df, test_df = add_w2v_features(train_df, valid_df, test_df)
        anime_embeddings_df, user_embeddings_df = fe.extract_nn_nfc(train_df, seed, 64, 128, 10, True)

        train_df = train_df.merge(user_embeddings_df, on='user_id', how='left')
        valid_df = valid_df.merge(user_embeddings_df, on='user_id', how='left')
        test_df = test_df.merge(user_embeddings_df, on='user_id', how='left')

        train_df = train_df.drop(['japanese_name'], axis=1)
        valid_df = valid_df.drop(['japanese_name'], axis=1)
        test_df = test_df.drop(['japanese_name'], axis=1)

        

        train_data_x, valid_data_x = train_df.drop(['score', 'fold'], axis=1), valid_df.drop(['score', 'fold'], axis=1)
        train_data_y, valid_data_y = train_df['score'], valid_df['score']

        # Perform masking on the 'user_id' and 'anime_id' columns
        rang = np.random.default_rng(fold)
        mask_user = rang.random(len(train_data_x)) < 0.05
        train_data_x.loc[mask_user, 'user_id'] = -1

        rang = np.random.default_rng(fold+1)
        mask_anime = rang.random(len(train_data_x)) < 0.05
        train_data_x.loc[mask_anime, 'anime_id'] = -1


        model = xgboost.XGBRegressor(**xgb_params)
        model.fit(train_data_x, train_data_y, 
                early_stopping_rounds=50, 
                eval_set=[(valid_data_x, valid_data_y)]
            , verbose=500)
        
        val_pred = model.predict(valid_data_x)
        valid_df['pred'] = val_pred
        rmse_score = np.sqrt(mean_squared_error(valid_data_y, val_pred))
        print('all valid rmse : ', rmse_score)

        score_list.append(rmse_score)

        valid_dfs = pd.concat([valid_df, valid_dfs], axis=0)
        test_pred = model.predict(test_df)
        tmp_test['score'] += test_pred
        
        if fold == 0:
            _, ax = plt.subplots(figsize=(16, 320))
            xgboost.plot_importance(model,
                            ax=ax,
                            importance_type='weight',
                            )
            plt.savefig(output_path+str(seed)+'_importance_plot.png')

    tmp_test['score'] = tmp_test['score'] / FOLD
    tmp_test.to_csv(output_path+str(seed)+'_seen_test.csv', index=False)
    print('all rmse : ',np.mean(score_list))
    valid_dfs.reset_index(drop=True).to_parquet(output_path+str(seed)+'_seen_pred.parquet')

if __name__ == '__main__':
    args = sys.argv
    if 2 == len(args):
        seed = int(args[1])
        main(seed)
    else:
        print('Incollect Argments')
