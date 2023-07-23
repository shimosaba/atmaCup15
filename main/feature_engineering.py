import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")
import scipy.sparse as sp

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold
from scipy.cluster.hierarchy import DisjointSet
import Levenshtein
from itertools import combinations
from transformers import (
    AutoTokenizer,
    AutoModel, 
    AutoConfig
)
from sklearn.decomposition import (
    TruncatedSVD,
    NMF,
)
from sklearn.pipeline import (
    make_pipeline, 
    make_union
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import _document_frequency
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec
from lightfm import LightFM
from lightfm.data import Dataset

def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    return df

def label_encoding(df, category_cols):
    le = LabelEncoder()
    for c in category_cols:
        le = le.fit(df[c])
        df[c] = le.transform(df[c])
    return df

def get_dummy_words(df, cols):
    for c in cols:
        # 列を','で分割し、フラットなリストに結合
        words = [word.strip() for col_list in df[c] for word in col_list.split(',')]
        # 重複する単語を削除
        unique_words = list(set(words))
        for w in unique_words:
            df[c+'_'+w] = df[c].apply(lambda x: 1 if w in x else 0)
    
    return df

def dummy_df(df):
    multilabel_cols = ["genres", "producers", "licensors", "studios"]
    multilabel_dfs = []
    for c in multilabel_cols:
        list_srs = df[c].map(lambda x: x.split(", ")).tolist()
        # MultiLabelBinarizerを使うと簡単に変換できるのでオススメです
        mlb = MultiLabelBinarizer()
        ohe_srs = mlb.fit_transform(list_srs)
        if c == "genres" or c == "licensors":
            # ユニーク数が多くないのでOne-hot表現のまま
            col_df = pd.DataFrame(ohe_srs, columns=[f"ohe_{c}_{name}" for name in mlb.classes_])
        else:
            # ユニーク数が多いので、SVDで次元圧縮する
            svd = TruncatedSVD(n_components=20)
            svd_arr = svd.fit_transform(ohe_srs)
            col_df = pd.DataFrame(
                svd_arr,
                columns=[f"svd_{c}_{ix}" for ix in range(20)]
            )
        multilabel_dfs.append(col_df)

    multilabel_df = pd.concat(multilabel_dfs, axis=1)

    df = pd.concat([df, multilabel_df], axis=1)

    return df


def target_encoding(df, SEED):
    cols = ['japanese_name', 'original_work_name', 'cluster']
    for c in cols:
        agg_df = df.groupby(c).agg({'score': ['sum', 'count']})

        ts = pd.Series(np.empty(df.shape[0]), index=df.index)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        for _, holdout_idx in folds.split(df, df[c]):
            # ホールドアウトする行を取り出す
            holdout_df = df.iloc[holdout_idx]
            # ホールドアウトしたデータで合計とカウントを計算する
            holdout_agg_df = holdout_df.groupby(c).agg({'score': ['sum', 'count']})
            # 全体の集計からホールドアウトした分を引く
            train_agg_df = agg_df - holdout_agg_df
            # ホールドアウトしたデータの平均値を計算していく
            oof_ts = holdout_df.apply(lambda row: train_agg_df.loc[row[c]][('score', 'sum')] \
                                                / (train_agg_df.loc[row[c]][('score', 'count')]) + 1, axis=1)
            # 生成した特徴量を記録する
            ts[oof_ts.index] = oof_ts

        ts.name = c + '_ts'
        df = df.join(ts)

    return df

# Function to split date into year, month, day
def split_date(date_str):
    if date_str is None or date_str == "Unknown":
        return [np.nan, np.nan, np.nan]
    parts = date_str.split()
    if len(parts) == 1:  # case when only year is available
        year = int(parts[0]) if parts[0].isdigit() else None
        return [year, np.nan, np.nan]
    elif len(parts) == 2:  # case when both year and month are available
        year = int(parts[1]) if parts[1].isdigit() else None
        return [year, parts[0], np.nan]
    elif len(parts) == 3:  # case when day, month, and year are available
        year = int(parts[2]) if parts[2].isdigit() else None
        day = int(parts[1][:-1]) if parts[1][:-1].isdigit() else None
        return [year, parts[0], day]

# Function to convert duration string to minutes
def duration_to_minutes(duration_str, episodes):
    if duration_str == "Unknown":
        return None
    duration_str = duration_str.replace(" per ep.", "")
    parts = duration_str.split()
    if len(parts) == 2:  # case when only minute is available
        minute = int(parts[0])
    elif len(parts) == 4:  # case when both hour and minute are available
        hour = int(parts[0])
        minute = int(parts[2])
        minute += hour * 60
    if episodes != "Unknown":
        minute *= int(episodes)
    return minute


# bert
class BertSequenceVectorizer:
    def __init__(self, model_name="bert-base-multilingual-uncased", max_len=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_config = AutoConfig.from_pretrained(self.model_name, num_labels=0)
        self.bert_model = AutoModel.from_pretrained(self.model_name, config=self.model_config)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = max_len

    def vectorize(self, sentence: str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy() # 0番目は [CLS] token, 768 dim の文章特徴量
        else:
            return seq_out[0][0].detach().numpy()
        

def add_bert(df):
    
    ## extract feature from text
    df['text'] = df['genres']+' '+df['japanese_name']+' '+df['type']+' '+df['producers']+' '+df['licensors']+' '+df['studios']+' '+df['aired']
    BSV = BertSequenceVectorizer(model_name="bert-base-multilingual-uncased", max_len=64)
    features = np.stack(df['text'].fillna("").map(lambda x: BSV.vectorize(x).reshape(-1)).values)

    ## clustering anime title
    kmeans = KMeans(n_clusters=100, max_iter=30, init="k-means++", n_init=30, random_state=42)
    cluster = kmeans.fit_predict(features)
    df['cluster'] = cluster

    pca = PCA(n_components=32)
    pca_features = pca.fit_transform(features)
    df = pd.concat([df, pd.DataFrame(pca_features).add_prefix('BERT_text')], axis=1)

    df = df.drop(['text'], axis=1)

    del BSV, features, kmeans
    gc.collect()

    return df

# Universal Sentence Encoder
def use(df, SEED):
    df['text'] = df['genres']+' '+df['japanese_name']+' '+df['type']+' '+df['producers']+' '+df['licensors']+' '+df['studios']+' '+df['aired']

    embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    features = np.stack(df["japanese_name"].fillna("").apply(lambda x: embedder(x).numpy().reshape(-1)).values)
    svd = TruncatedSVD(n_components=32, random_state=SEED)
    svd_features = svd.fit_transform(features)

    df = pd.concat([df, pd.DataFrame(svd_features[:len(df)]).add_prefix('USE_')], axis=1)

    df = df.drop(['text'], axis=1)
    del embedder
    gc.collect()
    return df

def get_sequence_tfidf(input_df, col, n_comp, SEED):
    vectorizer = make_pipeline(
        TfidfVectorizer(),
        make_union(
            TruncatedSVD(n_components=n_comp, random_state=SEED),
            NMF(n_components=n_comp, random_state=SEED),
        n_jobs=1)
    )
    sequences = input_df[col]
    X = vectorizer.fit_transform(sequences).astype(np.float32)
    cols = (
        [f'{col}_tfidf_svd_{i}' for i in range(n_comp)]
        + [f'{col}_tfidf_nmf_{i}' for i in range(n_comp)]
    )
    output_df = pd.DataFrame(X, columns=cols)
    return output_df

def add_w2v_features_without_score(df, SEED):

    anime_ids = df['japanese_name'].unique().tolist()
    user_anime_list_dict = {user_id: anime_ids.tolist() for user_id, anime_ids in df.groupby('user_id')['japanese_name']}

    title_sentence_list = df.groupby('user_id')['japanese_name'].apply(list).tolist()

    # ユーザごとにshuffleしたリストを作成
    shuffled_sentence_list = [random.sample(sentence, len(sentence)) for sentence in title_sentence_list]  ## <= 変更点

    # 元のリストとshuffleしたリストを合わせる
    train_sentence_list = title_sentence_list + shuffled_sentence_list

    # word2vecのパラメータ
    vector_size = 64
    random.seed(SEED)
    w2v_params = {
        "vector_size": vector_size,  ## <= 変更点
        "epochs": 10,
        "seed": SEED,
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
    user_factors_df.columns = ["user_id"] + [f"wo_score_user_factor_{i}" for i in range(vector_size)]
    item_factors_df.columns = ["japanese_name"] + [f"wo_score_item_factor_{i}" for i in range(vector_size)]

    df = df.merge(user_factors_df, on="user_id", how="left")
    df = df.merge(item_factors_df, on="japanese_name", how="left")

    return df

def extract_lightfm(df, seed):
    user_ids  = df.sort_values('user_id')['user_id'].tolist()
    anime_ids = df.sort_values('user_id')['anime_id'].tolist()

    dataset = Dataset()
    dataset.fit(users=user_ids, items=anime_ids)

    (interactions, weights) = dataset.build_interactions((user_id, anime_id) for user_id, anime_id in zip(user_ids, anime_ids))

    model = LightFM(loss='warp', random_state=seed, no_components=32)
    model.fit(interactions, epochs=30)

    user_representations = model.user_embeddings
    item_representations = model.item_embeddings
    user_mappings, _, item_mappings, _ = dataset.mapping()

    user_features_df = pd.DataFrame(user_representations).add_prefix('user_lightfm')
    item_features_df = pd.DataFrame(item_representations).add_prefix('anime_lightfm')
    inverse_user_mappings = {v: k for k, v in user_mappings.items()}
    inverse_item_mappings = {v: k for k, v in item_mappings.items()}

    user_features_df['user_id'] = [inverse_user_mappings[i] for i in range(user_representations.shape[0])]
    item_features_df['anime_id'] = [inverse_item_mappings[i] for i in range(item_representations.shape[0])]

    df = df.merge(user_features_df, on='user_id', how='left')
    df = df.merge(item_features_df, on='anime_id', how='left')

    return df

class AnimeDataset(Dataset):
    def __init__(self, user_ids, anime_ids, scores):
        self.user_ids = user_ids
        self.anime_ids = anime_ids
        self.scores = scores

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.anime_ids[idx], self.scores[idx]
class EmbOnlyCollabFNet(nn.Module):
    def __init__(self, num_users, num_animes, emb_size=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.anime_emb = nn.Embedding(num_animes, emb_size)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.anime_emb.weight)

    def forward(self, user_ids, anime_ids):
        user_emb = self.user_emb(user_ids)
        anime_emb = self.anime_emb(anime_ids)
        
        # Dot product between user and item embeddings
        scores = (user_emb * anime_emb).sum(dim=1)
        return scores
    
class CollabFNet(nn.Module):
    def __init__(self, n_users, n_animes, embed_dim, hidden_dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.anime_emb = nn.Embedding(n_animes, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, user_ids, anime_ids):
        user_embed = self.user_emb(user_ids)
        anime_embed = self.anime_emb(anime_ids)
        x = torch.cat([user_embed, anime_embed], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for user_ids, anime_ids, scores in dataloader:
        # Move data to device
        user_ids = user_ids.to(device)
        anime_ids = anime_ids.to(device)
        scores = scores.float().to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(user_ids, anime_ids)
        loss = criterion(outputs, scores)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Evaluation function
def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for user_ids, anime_ids, scores in dataloader:
            # Move data to device
            user_ids = user_ids.to(device)
            anime_ids = anime_ids.to(device)
            scores = scores.float().to(device)

            # Forward pass
            outputs = model(user_ids, anime_ids)
            loss = criterion(outputs, scores)

            total_loss += loss.item()

    return total_loss / len(dataloader)

def extract_nn_nfc(df, seed, emb, hidden, epoch, only_emb_model_f):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp = df.copy()
    # Create a label encoder object
    le_user = LabelEncoder()
    le_anime = LabelEncoder()

    # Fit and transform the user_id and anime_id to convert them into numeric values
    tmp['user_id'] = le_user.fit_transform(tmp['user_id'])
    tmp['anime_id'] = le_anime.fit_transform(tmp['anime_id'])
    

    # Get the number of unique users and anime
    num_users = tmp['user_id'].nunique()
    num_anime = tmp['anime_id'].nunique()

    train_dataset = AnimeDataset(
        tmp['user_id'].values,
        tmp['anime_id'].values,
        tmp['score'].values
    )
    val_dataset = AnimeDataset(
        tmp['user_id'].values,
        tmp['anime_id'].values,
        tmp['score'].values
    )

    batch_size = 128
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True,  
                                  num_workers=os.cpu_count(), 
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size,  
                                num_workers=os.cpu_count(), 
                                pin_memory=True)

    # Get the number of unique users and animes
    num_users = tmp['user_id'].nunique()
    num_animes = tmp['anime_id'].nunique()

    # Create the model
    if only_emb_model_f:
        model = EmbOnlyCollabFNet(num_users, num_animes, emb)
    else:
        model = CollabFNet(num_users, num_animes, emb, hidden)
    model = model.to(device)

    criterion = nn.MSELoss(reduce='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

    # Number of training epochs
    num_epochs = epoch

    # Train the model
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss = evaluate_epoch(model, val_dataloader, criterion, device)
        if epoch == num_epochs-1:
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    # Get anime embeddings
    anime_embeddings = model.anime_emb.weight.data.cpu().numpy()
    anime_embeddings_df = pd.DataFrame(anime_embeddings, columns=[f'nn_nfc_anime_feature_{i}' for i in range(anime_embeddings.shape[1])])
    anime_embeddings_df['anime_id'] = le_anime.inverse_transform(anime_embeddings_df.index)

    user_embeddings = model.user_emb.weight.data.cpu().numpy()
    user_embeddings_df = pd.DataFrame(user_embeddings, columns=[f'nn_nfc_user_feature_{i}' for i in range(user_embeddings.shape[1])])
    user_embeddings_df['user_id'] = le_user.inverse_transform(user_embeddings_df.index)

    return anime_embeddings_df, user_embeddings_df

def extract_lightfm_without_user(df, seed):
    user_ids  = df.sort_values('user_id')['user_id'].tolist()
    anime_ids = df.sort_values('user_id')['anime_id'].tolist()

    dataset = Dataset()
    dataset.fit(users=user_ids, items=anime_ids)

    (interactions, weights) = dataset.build_interactions((user_id, anime_id) for user_id, anime_id in zip(user_ids, anime_ids))

    model = LightFM(loss='warp', random_state=seed, no_components=32)
    model.fit(interactions, epochs=30)

    item_representations = model.item_embeddings
    user_mappings, _, item_mappings, _ = dataset.mapping()

    item_features_df = pd.DataFrame(item_representations).add_prefix('anime_lightfm')
    inverse_item_mappings = {v: k for k, v in item_mappings.items()}

    item_features_df['anime_id'] = [inverse_item_mappings[i] for i in range(item_representations.shape[0])]

    df = df.merge(item_features_df, on='anime_id', how='left')

    return df

# Levenshtein距離, jaro-winkler距離でシリーズものをまとめる
def get_original_work_name(df, threshold=0.3):
    _feature = df.japanese_name.tolist()
    _n = df.shape[0]

    _disjoint_set = DisjointSet(list(range(_n)))
    for i, j in combinations(range(_n), 2):
        if _feature[i] is np.nan or _feature[j] is np.nan:
            lv_dist, jw_dist = 0.5, 0.5
        else:
            lv_dist = 1 - Levenshtein.ratio(_feature[i], _feature[j])
            jw_dist = 1 - Levenshtein.jaro_winkler(_feature[i], _feature[j])
        _d = (lv_dist + jw_dist) / 2

        if _d < threshold:
            _disjoint_set.merge(i, j)

    _labels = [None] * _n
    for subset in _disjoint_set.subsets():
        label = _feature[list(subset)[0]]
        for element in subset:
            _labels[element] = label
    df["original_work_name"] = _labels

    return df

# 名寄せ
def convert_title(df):
    japanese_name_list = []
    for name in df['japanese_name'].values:
        #小文字化
        name = name.lower()
        name = name.replace('シーズン', 'season ')
        
        # タイトル日本語→英語
        name = name.replace('クラナド', 'clannad ')
        name = name.replace('ナルト', 'naruto ')
        name = name.replace('ハンターｘハンター', 'hunter×hunter ')
        name = name.replace('ワンピース', 'one piece ')

        japanese_name_list.append(name)

    df['japanese_name'] = japanese_name_list

    return df

def process_dataframe(df):
    # Split the 'aired' column into 'start_date' and 'end_date'
    df[['start_date', 'end_date']] = df['aired'].str.split(' to ', expand=True)

    # Split 'start_date' and 'end_date' into year, month, day
    df[['start_year', 'start_month', 'start_day']] = df['start_date'].apply(split_date).apply(pd.Series)
    df[['end_year', 'end_month', 'end_day']] = df['end_date'].apply(split_date).apply(pd.Series)

    # Map of month names to numbers
    month_to_num = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }

    # Convert 'start_month' and 'end_month' to numbers
    df['start_month'] = df['start_month'].map(month_to_num)
    df['end_month'] = df['end_month'].map(month_to_num)

    temp_df = df.rename(columns={'start_year': 'year', 'start_month': 'month', 'start_day': 'day'})
    df['start_date'] = pd.to_datetime(temp_df[['year', 'month', 'day']])
    df['aired_order'] = df['start_date'].rank()
    df['aired_cluster_order'] = df.sort_values('start_date').groupby('cluster').cumcount() + 1

    # Convert 'duration' to minutes per episode
    df['minutes_per_episode'] = df.apply(lambda row: duration_to_minutes(row['duration'], row['episodes']), axis=1)

    # Drop 'start_date', 'end_date', 'aired', and 'duration' columns
    df = df.drop(columns=['start_date', 'end_date', 'aired'])
    
    return df

def create_anime_df(anime, seed):
    anime['japanese_name_len'] = anime['japanese_name'].apply(lambda x: len(x))
    # 名寄せ
    anime = convert_title(anime)
    # シリーズものをクラスタリング
    anime = get_original_work_name(anime)
    # japanese_name feature engineering
    anime = add_bert(anime)
    anime = use(anime, seed)

    # TF IDS
    tf_ids = get_sequence_tfidf(anime, 'japanese_name', 16, seed)
    anime = pd.concat([anime.reset_index(drop=True), tf_ids], axis=1)

    # one hot encoding
    anime = dummy_df(anime)

    # label encoding
    label_cols = ['type', 'rating', 'source', 'genres', 'original_work_name']
    anime = label_encoding(anime, label_cols)    

    # count encoding
    for c in label_cols:
        vc = anime[c].value_counts()
        anime[c+'_count'] = anime[c].map(vc)

    anime = process_dataframe(anime)

    # anime['episodes'] = anime['episodes'].apply(lambda x: int(x) if 'Unknown' != x else np.nan)
    epi_list = []
    for epi in anime['episodes'].values:
        if epi != 'Unknown':
            epi_list.append(int(epi))
        else:  
            epi_list.append(np.nan)
    anime['episodes'] = epi_list

    anime = label_encoding(anime, ['duration'])

    anime['drop_rate'] = anime['dropped'] / anime['members']
    anime['completed_rate'] = anime['completed'] / anime['members']
    anime['watching_rate'] = anime['watching'] / anime['members']
    anime['plan_to_watch_rate'] = anime['plan_to_watch'] / anime['members']    

    anime = anime.drop(['licensors', 'producers', 'studios'], axis=1)
    return anime


def similer_user_anime(df, anime, SEED):
    # Find the top 10 animes by score for each user
    top_animes_per_user = df[df['class']==0].groupby('user_id').apply(lambda x: x.nlargest(20, 'score')['anime_id'].tolist())
    top_animes_per_user_df = top_animes_per_user.to_frame().reset_index()
    top_animes_per_user_df.columns = ['user_id', 'top_animes']
    random.seed(SEED)

    w2v_params = {
        "vector_size": 64,
        "epochs": 10,
        "seed": SEED,
        "min_count": 1,
        "workers": 1, 
        "window": 15
    }

    title_sentence_list = list(top_animes_per_user_df['top_animes'].values)
    # ユーザごとにshuffleしたリストを作成
    shuffled_sentence_list = [random.sample(sentence, len(sentence)) for sentence in title_sentence_list]  ## <= 変更点
    # 元のリストとshuffleしたリストを合わせる
    train_sentence_list = title_sentence_list + shuffled_sentence_list
    # word2vecのモデル学習
    model = word2vec.Word2Vec(train_sentence_list, **w2v_params)
    
    similer_name_list = []
    similer_score_list = []
    for name in anime['anime_id'].values:
        if name in model.wv.key_to_index.keys():
            similer_name_list.append([title[0] for title in model.wv.most_similar(name)])
            similer_score_list.append([title[1] for title in model.wv.most_similar(name)])
        else:
            similer_name_list.append([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
            similer_score_list.append([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])

    anime = pd.concat([anime[['anime_id']], pd.DataFrame(similer_name_list).add_prefix('similer_name')], axis=1)
    anime = pd.concat([anime, pd.DataFrame(similer_score_list).add_prefix('similer_score')], axis=1)
    df = df.merge(anime, on='anime_id', how='left')

    return df

def similer_encoding(df, anime):
    nan_df = pd.DataFrame(index=[], columns=anime.columns)
    transaction = [np.nan] * len(anime.columns)
    record = pd.Series(transaction, index=nan_df.columns)
    nan_df.loc[len(nan_df)] = record
    anime = pd.concat([anime, nan_df], axis=0)
    le = LabelEncoder()
    le = le.fit(anime['anime_id'])
    df['anime_id'] = le.transform(df['anime_id'])
    for i in range(10):
        df['similer_name'+str(i)] = le.transform(df['similer_name'+str(i)])
    return df

def remove_anime_id(df, anime):
    anime_ids = df['anime_id'].unique()
    anime = anime[anime['anime_id'].isin(anime_ids)].reset_index(drop=True)
    return anime

def seen_add_feature(df,anime, seed):
    anime = remove_anime_id(df, anime)
    df = reduce_mem_usage(df)
    print('remove mem usage')
    df = extract_lightfm(df, seed)

    anime = create_anime_df(anime, seed)
    print('end create anime')    
    df = df.merge(anime, on='anime_id', how='left')

    # similer_user_anime
    df = similer_user_anime(df, anime, seed)

    temp_df = df.rename(columns={'start_year': 'year', 'start_month': 'month', 'start_day': 'day'})
    df['start_date'] = pd.to_datetime(temp_df[['year', 'month', 'day']])
    df['user_watch_rank'] = df.sort_values('start_date').groupby('user_id').cumcount() + 1
    df['user_type_rank'] = df.sort_values('start_date').groupby(['user_id','type']).cumcount() + 1
    agg_df = df.groupby(['user_id', 'type'])\
        .agg({'start_year':['min']})['start_year'].reset_index().rename(columns={'min':'user_type_min_year'})
    df = df.merge(agg_df, on=['user_id', 'type'], how='left')

    df = target_encoding(df, seed)
    print('end target encoding')

    # agg
    vc = df['japanese_name'].value_counts()
    df['japanese_name_count'] = df['japanese_name'].map(vc)
    vc = df['user_id'].value_counts()
    df['user_id_count'] = df['user_id'].map(vc)
    df['total_episode_min'] = df['episodes'] * df['minutes_per_episode']

    agg_df = df.groupby(['genres', 'type'])['user_id'].count().reset_index().rename(columns={'user_id':'genres_type_count'})
    df = df.merge(agg_df, on=['genres', 'type'], how='left')

    num_list = ['members', 'watching', 'completed', 'on_hold', 
                'dropped', 'plan_to_watch', 'minutes_per_episode', 
                'total_episode_min', 'start_year']
    agg_list = ['type', 'episodes', 'source', 'rating', 'genres', 'original_work_name']
    for a in agg_list:
        for n in num_list:
            agg_df = df.groupby(['user_id', a]).agg({n:['mean','median','std','skew']})[n].reset_index()\
                .rename(columns={'mean':'user_'+a+'_'+n+'_mean', 
                                 'median':'user_'+a+'_'+n+'_median', 
                                 'std':'user_'+a+'_'+n+'_std',
                                 'skew':'user_'+a+'_'+n+'_skew'
                                 })
            
            agg_df = df.groupby(['user_id', a])[n].count().reset_index().rename(columns={n:'user_'+a+'_'+n+'_count'})
            df = df.merge(agg_df, on=['user_id', a], how='left')

            agg_df = df.groupby([a]).agg({n:['mean','median','std','skew']})[n].reset_index()\
                .rename(columns={'mean':a+'_'+n+'_mean', 
                                 'median':a+'_'+n+'_median', 
                                 'std':a+'_'+n+'_std',
                                 'skew':a+'_'+n+'_skew'
                                 })
            
            agg_df = df.groupby([a])[n].count().reset_index().rename(columns={n:a+'_'+n+'_count'})
            df = df.merge(agg_df, on=[a], how='left')

    print('end agg feature')

    ids = df[df['class']==0][['user_id','anime_id']].rename(columns={'user_id':'original_user_id','anime_id':'original_anime_id'})

    # # label encoding
    label_cols = ['user_id']
    df = label_encoding(df, label_cols)

    # similer name encoding
    df = similer_encoding(df, anime)

    df = df.drop(['start_date', 'japanese_name_len', 'cluster_ts', 'duration', 'completed_rate'], axis=1)

    print(df.shape)

    ids['user_id'] = df[df['class']==0]['user_id']
    ids['anime_id'] = df[df['class']==0]['anime_id']

    df = reduce_mem_usage(df)

    torch.cuda.empty_cache()
    tf.keras.backend.clear_session()

    return df, ids

def unseen_add_feature(df,anime, seed):
    anime = remove_anime_id(df, anime)
    df = reduce_mem_usage(df)
    df = extract_lightfm_without_user(df, seed)

    anime = create_anime_df(anime, seed)
    print('end create anime')
    df = df.merge(anime, on='anime_id', how='left')

    # similer_user_anime
    df = similer_user_anime(df, anime, seed)

    temp_df = df.rename(columns={'start_year': 'year', 'start_month': 'month', 'start_day': 'day'})
    df['start_date'] = pd.to_datetime(temp_df[['year', 'month', 'day']])
    df['user_watch_rank'] = df.sort_values('start_date').groupby('user_id').cumcount() + 1
    df['user_type_rank'] = df.sort_values('start_date').groupby(['user_id','type']).cumcount() + 1
    df['user_cluster_rank'] = df.sort_values('start_date').groupby(['user_id','cluster']).cumcount() + 1
    df['user_original_work_name_rank'] = df.sort_values('start_date').groupby(['user_id','original_work_name']).cumcount() + 1

    agg_df = df.groupby(['user_id', 'type']).agg({'start_year':['max']})['start_year'].reset_index().rename(columns={'max':'user_type_max_year'})
    df = df.merge(agg_df, on=['user_id', 'type'], how='left')

    df = target_encoding(df, seed)
    print('end target encoding')

    # agg
    vc = df['japanese_name'].value_counts()
    df['japanese_name_count'] = df['japanese_name'].map(vc)
    vc = df['user_id'].value_counts()
    df['user_id_count'] = df['user_id'].map(vc)
    df['total_episode_min'] = df['episodes'] * df['minutes_per_episode']

    agg_df = df.groupby(['genres', 'type'])['user_id'].count().reset_index().rename(columns={'user_id':'genres_type_count'})
    df = df.merge(agg_df, on=['genres', 'type'], how='left')
    
    agg_df = df[df['type']=='TV'].groupby(['user_id','start_year'])['anime_id'].count().reset_index().rename(columns={'anime_id':'TV_user_start_count'})
    df = df.merge(agg_df, on=['user_id','start_year'], how='left')

    num_list = ['members', 'watching', 'completed', 
                'on_hold', 'dropped', 'plan_to_watch', 
                'minutes_per_episode', 'total_episode_min', 'start_year']
    agg_list = ['type', 'episodes', 'source', 'rating', 'genres', 'original_work_name']
    for a in agg_list:
        for n in num_list:
            agg_df = df.groupby(['user_id', a]).agg({n:['mean','median','std','skew']})[n].reset_index()\
                .rename(columns={'mean':'user_'+a+'_'+n+'_mean', 
                                 'median':'user_'+a+'_'+n+'_median', 
                                 'std':'user_'+a+'_'+n+'_std',
                                 'skew':'user_'+a+'_'+n+'_skew'
                                 })
            
            agg_df = df.groupby(['user_id', a])[n].count().reset_index().rename(columns={n:'user_'+a+'_'+n+'_count'})
            df = df.merge(agg_df, on=['user_id', a], how='left')

            agg_df = df.groupby([a]).agg({n:['mean','median','std','skew']})[n].reset_index()\
                .rename(columns={'mean':a+'_'+n+'_mean', 
                                 'median':a+'_'+n+'_median', 
                                 'std':a+'_'+n+'_std',
                                 'skew':a+'_'+n+'_skew'
                                 })
            
            agg_df = df.groupby([a])[n].count().reset_index().rename(columns={n:a+'_'+n+'_count'})
            df = df.merge(agg_df, on=[a], how='left')

    print('end agg feature')

    ids = df[df['class']==0][['user_id','anime_id']].rename(columns={'user_id':'original_user_id','anime_id':'original_anime_id'})

    # label encoding
    label_cols = ['user_id']
    df = label_encoding(df, label_cols)

    # similer name encoding
    df = similer_encoding(df, anime)
    df = add_w2v_features_without_score(df, seed)

    df = df.drop(['start_date', 'original_work_name_ts', 'drop_rate', 'watching_rate', 'plan_to_watch_rate'], axis=1)

    print(df.shape)

    ids['user_id'] = df[df['class']==0]['user_id']
    ids['anime_id'] = df[df['class']==0]['anime_id']

    df = reduce_mem_usage(df)

    return df, ids
