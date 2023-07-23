import pandas as pd
import numpy as np


input_path = '../../input/'


def main():
    test = pd.read_csv(input_path+'test.csv')
    sub = pd.read_csv(input_path+'sample_submission.csv')
    print(test.shape)

    sub['score'] = 0

    # seen_test_42 = pd.read_csv('./seen/42_seen_test.csv').reset_index(drop=True)
    # unseen_test_42 = pd.read_csv('./unseen/42_unseen_test.csv').reset_index(drop=True)
    # pred_test_42 = pd.concat([seen_test_42, unseen_test_42], axis=0).reset_index(drop=True)
    # pred_test_42 = pred_test_42.rename(columns={'score':'score_42'})
    # print(pred_test_42.shape)

    # seen_test_21 = pd.read_csv('./seen/21_seen_test.csv').reset_index(drop=True)
    # unseen_test_21 = pd.read_csv('./unseen/21_unseen_test.csv').reset_index(drop=True)
    # pred_test_21 = pd.concat([seen_test_21, unseen_test_21], axis=0).reset_index(drop=True)
    # pred_test_21 = pred_test_21.rename(columns={'score':'score_21'})
    # print(pred_test_21.shape)

    # pred = pd.merge(pred_test_42, pred_test_21, on=['user_id', 'anime_id'], how='left')
    # pred['score'] = (pred['score_42']+pred['score_21']) / 2

    # test = test.merge(pred[['user_id', 'anime_id', 'score']], on=['user_id', 'anime_id'], how='left')
    # sub['score'] = test['score']
    
    # sub['score'] = sub['score'].apply(lambda x: np.clip(x, 1, 10))
    # sub['score'] = sub['score'].round(2)
    # sub.to_csv('submission.csv', index=False)

    # seen_test_21 = pd.read_csv('./seen/21_seen_test.csv').reset_index(drop=True)
    # unseen_test_21 = pd.read_csv('./unseen/21_unseen_test.csv').reset_index(drop=True)
    # pred_test_21 = pd.concat([seen_test_21, unseen_test_21], axis=0).reset_index(drop=True)
    # print(pred_test_21.shape)

    # test = test.merge(pred_test_21[['user_id', 'anime_id', 'score']], on=['user_id', 'anime_id'], how='left')
    # sub['score'] = test['score']
    
    # sub['score'] = sub['score'].apply(lambda x: np.clip(x, 1, 10))
    # sub['score'] = sub['score'].round(2)
    # sub.to_csv('submission_seed21.csv', index=False)


    seen_test_42 = pd.read_csv('./seen/42_seen_test.csv').reset_index(drop=True)
    unseen_test_42 = pd.read_csv('./unseen/42_unseen_test.csv').reset_index(drop=True)
    pred_test_42 = pd.concat([seen_test_42, unseen_test_42], axis=0).reset_index(drop=True)
    print(pred_test_42.shape)

    test = test.merge(pred_test_42[['user_id', 'anime_id', 'score']], on=['user_id', 'anime_id'], how='left')
    sub['score'] = test['score']
    
    sub['score'] = sub['score'].apply(lambda x: np.clip(x, 1, 10))
    sub['score'] = sub['score'].round(2)
    
    sub.to_csv('submission_seed42.csv', index=False)

if __name__ == '__main__':
    main()