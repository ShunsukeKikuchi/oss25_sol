# TaskC: Keypoint Detector 学習・評価ガイド（最新版）

本ドキュメントは、TaskCのキーポイント検出器（KPNet）の学習〜推論〜評価までをまとめた手順書です。モデルはConvNeXtベースのFPN＋Heatmap/Visibilityヘッドに加え、オプションで6チャネル入力（RGB + Optical Flow(u,v) + Foreground）と補助セグ損失（Dice+Focal）および時間一貫性損失をサポートします。

---

## 1. データ前提・構成
- 学習画像（train）: `data/train/frames/<video>_frame_<id>.png`
- 学習ラベル（train MOT）: `data/train/mot/<video>_frame_<id>.txt`（各行にxyvを含む）
- 検証画像（val）: `data/val/frames/<video>/<video>_frame_<id>.png`
- 検証ラベル（val MOT 集約）: `data/val/mot/<video>.txt`
- HOTA-KP評価用GT: `artifacts/kp_eval/data/gt/<video>.txt`（用意済みを想定）

（Tips）固定視点前提のため、前フレームとの差分から簡易前景マスクが有効です。OF/前景はデータセット内で自動生成します（OpenCVが必要）。

---

## 2. 環境セットアップ
```
source .venv/bin/activate
pip install -r requirements/training.txt  # or requirements/common.txt 等
pip install opencv-python-headless       # OF/前景の生成に必要
```

---

## 3. SAM2による疑似マスク生成（任意・補助学習用）
疑似マスク（6クラス: 背景+ツール5種+手2種）は `artifacts/seg_pseudo6/<VIDEO>/*.png` に保存します。

例（10fpsに事前ダウンサンプル→伝播・保存）:
```
python scripts/run_sam2_propagate.py \
  --video A36O --fps 10 \
  --config configs/sam2.1/sam2.1_hiera_b+.yaml \
  --ckpt sam2/checkpoints/sam2.1_hiera_base_plus.pt \
  --device cuda:0

# 複数動画（並列）
cat splits/oss25_5fold_v2/all_videos.txt | xargs -n1 -P4 -I{} bash -lc \
  'python scripts/run_sam2_propagate.py --video {} --fps 10 --config configs/sam2.1/sam2.1_hiera_b+.yaml --ckpt sam2/checkpoints/sam2.1_hiera_base_plus.pt --device cuda:0'
```

---

## 4. 学習（推奨レシピ）

学習は2段階（Stage 1→Stage 2）を推奨します。

### 4.1 Stage 1（セグ補助＋時間一貫性あり）
- 入力: 6ch（RGB + Flow(u,v) + Foreground）
- Heatmapサイズ: 196（細かい量子化に有利）
- 補助損失: セグ（Dice+Focal）、時間一貫性（HM/Coord）を少量

例:
```
PYTHONPATH=src python src/oss25/train_kpnet.py \
--data-root ./data \
--image-size 540,960 \
--in-chans 6 \
--batch-size 8 --epochs 60 --lr 1e-4 --weight-decay 1e-4 \
--hm-pos-weight 80 \
--coord-loss-weight 0.2 \
--hm-consist-weight 0.05 --coord-consist-weight 0.05 \
--aug-rotate-deg 0 \
--seg-loss-weight 0.15 \
--pseudo6-root artifacts/seg_pseudo6 \
--out-dir artifacts/exp2_pretrain \
--device cuda \
--dump-seg-val --dump-seg-interval 1 --dump-seg-limit 1000 --dump-seg-alpha 0.9 \
--resize-only \
--val-hota 
```

チェックポイントは `artifacts/kpnet/{last.pt,best.pt}` に保存されます。

### 4.2 Stage 2（ファインチューニング）
- 目的: 補助損失を切って、キーポイント精度を微調整
- 推奨: `--seg-loss-weight 0.0`, `--aug-rotate-deg 0`（安定化）、学習率据え置き〜微下げ

例（Stage 1の最終から再開）:
```
PYTHONPATH=src python src/oss25/train_kpnet.py \
  --resume artifacts/exp2_pretrain/last.pt \
  --data-root ./data \
  --image-size 540,960 \
  --in-chans 6 \
  --batch-size 8 --epochs 20 \
  --hm-pos-weight 80 \
  --coord-loss-weight 0.05 \
  --hm-consist-weight 0.05 --coord-consist-weight 0.05 \
  --aug-rotate-deg 0 \
  --seg-loss-weight 0.0 \
  --pseudo6-root artifacts/seg_pseudo6 \
  --out-dir artifacts/exp2_finetune \
  --device cuda \
  --dump-seg-val --dump-seg-interval 1 --dump-seg-limit 1000 \
  --resize-only \
  --val-hota \
  --val-hota-interval 1
```

（補足）`--init <ckpt>` でモデルのみ読み込み（optimizerは新規）、`--resume <ckpt>` で学習全体を再開できます。

---

## 5. 推論

学習時の `--heatmap-size` と同じ値を推論でも必ず指定してください（例: 196）。入力チャネル数はckptから自動推定します（`--in-chans`を指定してもckpt優先）。

### 5.1 Detector-only（フレーム独立）
```
PYTHONPATH=src python -m oss25.infer_kpnet_cotracker \
  --data-root ./data \
  --ckpt artifacts/exp1/best.pt \
  --out-dir artifacts/kp_eval/data/trackers \
  --device cuda \
  --image-size 384 --heatmap-size 196
```

### 5.2 Detector + CoTracker（Kフレーム毎に再検出）
```
PYTHONPATH=src python -m oss25.infer_kpnet_with_cotracker \
  --data-root ./data \
  --ckpt artifacts/kpnet/best.pt \
  --out-dir artifacts/kp_eval/data/trackers_cotracker \
  --device cuda \
  --image-size 384 --heatmap-size 196 \
  --k 10 --grid-size 20 --vis-th 0.5
```

### 5.3 Detector + KLT（軽量ブリッジ）
```
PYTHONPATH=src python -m oss25.infer_kpnet_klt \
  --data-root ./data \
  --ckpt artifacts/kpnet/best.pt \
  --out-dir artifacts/kp_eval/data/trackers_klt \
  --device cuda \
  --image-size 384 --heatmap-size 196 \
  --k 10
```

---

## 6. 評価（HOTA-KP）
TrackEvalの公式スクリプトを使用します。以下は例です。
```
python TrackEval/scripts/run_mot_challenge_kp.py \
  --GT_FOLDER artifacts/kp_eval/data/gt \
  --TRACKERS_FOLDER artifacts/kp_eval/data/trackers \
  --OUTPUT_FOLDER artifacts/kp_eval/out_det \
  --TRACKERS_TO_EVAL oss25_det
```

（注）`--TRACKERS_FOLDER` を CoTracker/KLT の出力フォルダに切り替えて比較してください。

---

## 7. 可視化（GT–予測の対応線付き）
対応するGT点と予測点の間に直線を描き、対応関係を分かりやすく表示します。
```
PYTHONPATH=src python -m oss25.visualize_val_kp \
  --data-root ./data \
  --gt-folder artifacts/kp_eval/data/gt \
  --pred artifacts/kp_eval/data/trackers/E66F_pred.txt \
  --video E66F \
  --out artifacts/kp_eval/vis/E66F_refine \
  --limit 12
```

---

## 8. 代表的なCLI引数
- `--image-size`: 入力の正方形リサイズ解像度（既定384）
- `--heatmap-size`: ヒートマップ解像度（例: 128/196）。学習と推論で必ず一致
- `--in-chans`: 入力チャネル数（3=RGB, 6=RGB+flow+fg）。推論はckptから自動推定
- `--hm-pos-weight`: Heatmap BCEの正例強調（クラス不均衡対策）
- `--coord-loss-weight`: 座標L1の重み
- `--seg-loss-weight`: セグ補助（Dice+Focal）の重み（0で無効）
- `--hm-consist-weight`, `--coord-consist-weight`: 時間一貫性損失の重み（0で無効）
- `--aug-rotate-deg`: 画像・GT・擬似マスクを一括回転する度数（trainのみ）
- `--resume` / `--init`: 再開（optimizer含む）/ 初期化（モデル重みのみ）

---

## 9. トラブルシューティング
- HOTAが0になる / `unexpected=decoder.*` が出る
  - 学習と推論の `--heatmap-size` を一致させてください（例: 196）。解像度が違うとデコーダ段数がズレ、重みが適用されません
- 6ch学習済ckptを3chで推論してしまう
  - 推論スクリプトはckptから自動で`in_chans`を推定しますが、OpenCVが無効だとflow/fgがゼロ埋めになり精度低下します。`pip install opencv-python-headless`
- データローダが重い
  - OF計算がCPU負荷になる場合は `--num-workers` を下げる、あるいはOF/前景を別途前計算・キャッシュする運用をご検討ください

---

## 10. 実装の要点（参考）
- モデル: `src/oss25/models/kpnet.py`（ConvNeXt + FPN + Heatmap/Vis/Seg）
- データ: `src/oss25/task3_kp_dataset.py`（OF/前景の生成、回転、擬似マスク取り込み）
- 学習/損失: `src/oss25/train_kpnet.py`（BCE+coord L1、Dice+Focal、HM/Coord時間一貫性、resume/init）
- 推論: `src/oss25/infer_kpnet_cotracker.py`, `src/oss25/infer_kpnet_with_cotracker.py`, `src/oss25/infer_kpnet_klt.py`
- 可視化: `src/oss25/visualize_val_kp.py`
