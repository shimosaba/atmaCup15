# atmaCup #15　16th place solution

解法は[atmaCup#15のディスカッション](https://www.guruguru.science/competitions/21/discussions/d4c0cdc2-8214-4cf9-8c15-c534a8ffc202/)に投稿しました。

## 実行方法
ensembleもまとめて実行
```
sh exep.sh
```
seed指定してそれぞれ実行
```
python3 seen_exp.py 42
python3 unseen_exp.py 42
python3 make_submission.py
```

## 環境
- Windows11+WSL2+Docker
- CPU
  - AMD Ryzen 5 5600 6-Core Processor
- RAM
  - 48GB
- GPU
  - NVIDIA RTX A4000 


## 上位解法のアイデア
- 作品に関係する人数（"members", "watching", "completed", "on_hold", "dropped", "plan_to_watch"）
  - userごとにgroupbyした特徴量をさらにanimeごとにgroupbyするなど、2hop,3hop先の情報を特徴量として取得できるように次のように工夫
    - userごとに特徴量を集約
    - アニメに対してそれを見たユーザーが複数存在するので、そのユーザー達の集約結果をさらに集約してアニメの特徴量を作る
  - その anime_id と類似した作品をそのユーザーがどれだけ見ているのかの情報（編集距離の統計量を取る）
  - 特徴量的には一番効いていそう
- [GNN](https://zenn.dev/kami/articles/83c2daff760f5d)
