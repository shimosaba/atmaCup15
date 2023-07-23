import sys
import numpy as np
import pandas as pd
import random
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_squared_error

import xgboost

import feature_engineering as fe

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

input_path = '../../input/'
output_path = './unseen/'

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

    # filter unseen test user
    seen_user = train['user_id'].unique()
    tmp_test = test.copy()
    tmp_test = tmp_test[~tmp_test['user_id'].isin(seen_user)].reset_index(drop=True)
    tmp_test['score'] = 0

    # feature engineering
    train['class'] = 0
    tmp_test['class'] = 1
    df = pd.concat([train, tmp_test])
    df, ids = fe.unseen_add_feature(df, anime, seed)

    ids.to_parquet(output_path+str(seed)+'_ids.parquet')
    df.to_parquet(output_path+str(seed)+'_featured_df.parquet')
    df = pd.read_parquet(output_path+str(seed)+'_featured_df.parquet')
    ids = pd.read_parquet(output_path+str(seed)+'_ids.parquet')

    df = df.drop(['japanese_name'], axis=1)

    train = df[df['class']==0]
    test = df[df['class']==1]

    train = train.drop(['class'], axis=1)
    train['score'] = train['score'].astype(int)
    test = test.drop(['score','class'], axis=1)

    FOLD = 5
    skf = StratifiedGroupKFold(n_splits=FOLD, random_state=seed, shuffle=True)
    for fold_id, (_, valid_idx) in enumerate(skf.split(train, train["score"], groups=train['user_id'])):
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
                'subsample': 0.7, 
                'colsample_bytree': 0.7, 
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
        
        agg_df = train_df.groupby('anime_id')['score'].median().reset_index().rename(columns={'score':'anime_score_median'})
        train_df = train_df.merge(agg_df, on='anime_id', how='left')
        valid_df = valid_df.merge(agg_df, on='anime_id', how='left')
        test_df = test_df.merge(agg_df, on='anime_id', how='left')        

        anime_embeddings_df, user_embeddings_df = fe.extract_nn_nfc(train_df, seed, 32, 64, 5, True)
        train_df = train_df.merge(anime_embeddings_df, on='anime_id', how='left')
        valid_df = valid_df.merge(anime_embeddings_df, on='anime_id', how='left')
        test_df = test_df.merge(anime_embeddings_df, on='anime_id', how='left')

        train_data_x, valid_data_x = train_df.drop(['score', 'fold'], axis=1), valid_df.drop(['score', 'fold'], axis=1)
        train_data_y, valid_data_y = train_df['score'], valid_df['score']

        # Perform masking on the 'user_id' and 'anime_id' columns
        rang = np.random.default_rng(fold)
        mask_anime = rang.random(len(train_data_x)) < 0.05
        train_data_x.loc[mask_anime, 'anime_id'] = -1
        
        rang = np.random.default_rng(fold+1)
        mask_anime = rang.random(len(train_data_x)) < 0.05
        train_data_x.loc[mask_anime, 'anime_score_median'] = -1

        model = xgboost.XGBRegressor(**xgb_params)
        model.fit(train_data_x.drop(['user_id'], axis=1), train_data_y, 
                early_stopping_rounds=50, 
                eval_set=[(valid_data_x.drop(['user_id'], axis=1), valid_data_y)]
            , verbose=500)
        
        val_pred = model.predict(valid_data_x.drop(['user_id'], axis=1))
        valid_df['pred'] = val_pred
        rmse_score = np.sqrt(mean_squared_error(valid_data_y, val_pred))
        print('all valid rmse : ', rmse_score)

        score_list.append(rmse_score)

        valid_dfs = pd.concat([valid_df, valid_dfs], axis=0)
        test_pred = model.predict(test_df.drop(['user_id'], axis=1))
        tmp_test['score'] += test_pred
        
        if fold == 0:
            _, ax = plt.subplots(figsize=(16, 320))
            xgboost.plot_importance(model,
                            ax=ax,
                            importance_type='weight'
                            )
            plt.savefig(output_path+str(seed)+'_importance_plot.png')

    tmp_test['score'] = tmp_test['score'] / FOLD
    tmp_test.to_csv(output_path+str(seed)+'_unseen_test.csv', index=False)
    print('all rmse : ',np.mean(score_list))
    valid_dfs.reset_index(drop=True).to_parquet(output_path+str(seed)+'_unseen_pred.parquet')

if __name__ == '__main__':
    args = sys.argv
    if 2 == len(args):
        seed = int(args[1])
        main(seed)
    else:
        print('Incollect Argments')
