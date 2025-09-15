# OSS 2025 Challenge 要件サマリ（整形版）

本ドキュメントは `OSS2025_pub.md` をもとに、実装観点で読みやすく要点整理したものです。

## 1. 概要
- 課題: 開腹（オープン）縫合の熟練度評価（動画ベース）。
- 提出対象タスク:
  - Task 1: GRS（Global Rating Score）4クラス分類（0–3）。
  - Task 2: OSATS 8カテゴリのスコア（各0–4）予測。
  - Task 3: 手と器具のキーポイント追跡（MOTChallenge 変形フォーマット）。
- 目的: 客観的・自動的なスキル評価を提供し、訓練を支援。

## 2. タイムライン（GMT基準）
- 2025-05-01: Web/登録オープン
- 2025-06-13: 学習データ公開
- 2025-07-02: バリデーションデータ公開
- 2025-07-05: 評価スクリプト公開
- 2025-08-01: 評価開始
- 2025-09-15 23:59: 最終提出締切
- 2025-09-17 23:59: Write-up 提出締切
- 2025-09-27: Challenge Day（DCC1-2F-209-211, 11:45 AM GMT+9）

## 3. データ概要
- 共通: 動画は俯瞰視点、約5分、被験者は学生やレジデント。各動画に3名の評価者。
- Task 1/2 用:
  - ラベル Excel: `OSATS_MICCAI_trainset.xlsx`
  - 項目例: `VIDEO, OSATS_*8種, GLOBARATINGSCORE (8–40)` など。
  - 注意: `GROUP` は分類に使用しない旨の注記あり。
- Task 3 用（Tracking）:
  - 学習: 1分クリップから1 fpmでサンプリング。手/器具のセグメンテーションマスクとKP（手6点, 器具3点）。
  - バリデーション: 1 fps、KPのみ。学習とディレクトリ構造が異なる点に注意。
  - マスク注意: 被覆オブジェクトはKPあり・マスク無しの可能性あり。針は最大2本（`needle1` サフィックス等）。

### Task 3 ディレクトリ例
- Train:
  - `train/frames/<video>_frame_<id>.png`
  - `train/masks/<video>_frame_<id>_<tool>[suffix]_mask.png`
  - `train/mot/<video>_frame_<id>.txt`
- Val（構造が異なる）:
  - `val/frames/<video>/<video>_frame_<id>.png`
  - `val/mot/<video>_frame_<id>.txt`

### キー情報（Task 3）
- セグメンテーションクラス: 0:左手, 1:右手, 2:ハサミ, 3:ピンセット, 4:ニードルホルダ, 5:針。
- 手KP: 親指/中/人差し/薬/小指/手背（合計6点）— 実データで確認済み。
- 器具KP: いずれも3点（例: Scissors: 左/右/ジョイント）。
- Visibility/Confidence: 0:枠外, 1:隠蔽, 2:可視。実データは `xyv` 形式（提出仕様の `c` はこの可視性指標と整合）。

## 4. 入出力仕様（提出物の出力フォーマット）
- 共通: 入力は動画ディレクトリ（mp4）。出力はCSV（Task 3は動画ごとファイル）。
- Task 1（GRS 4クラス）:
  - 出力CSV（ヘッダ必須）: `VIDEO, GRS`
  - クラス定義（参考）: 0: 8–15, 1:16–23, 2:24–31, 3:32–40。
- Task 2（OSATS 8カテゴリ）:
  - 出力CSV（ヘッダ必須）: `VIDEO, OSATS_RESPECT, OSATS_MOTION, OSATS_INSTRUMENT, OSATS_SUTURE, OSATS_FLOW, OSATS_KNOWLEDGE, OSATS_PERFORMANCE, OSATSFINALQUALITY`
- Task 3（Tracking, MOTChallenge-modified）:
  - 動画ごとにヘッダ無しのTXT/CSV: `Frame \t Track ID \t Class ID \t Bbox xywh \t KP xyc`
  - `Bbox xywh` は任意または `-1` 可、`KP xyc` は `(x, y, conf)` をコンマ区切りで連結。

## 5. 評価
- 共通: チームの最終提出のみ評価対象。フル提出が必須。
- Task 1/2: F1（Dice）と Expected Cost を使用。評価者平均で集計。メトリクス順位を平均。
- Task 3: HOTA（KP版）。リポジトリの `scripts/run_mot_challenge_kp.py`（分岐 `devel-kp`）を参照。

## 6. 技術要件・ルール
- Docker はオフライン実行（`--network none`）。ユーザ操作無しの完全自動。
- データのバリデーションセットでの学習は禁止（Task 3）。
- 不正（評価ラベル漏洩、サーバ悪用等）は失格。
- 主催者所属は参加・掲載可だが受賞対象外。

## 7. Docker 実行仕様
- 複数タスク対応コンテナ例:
  ```
  docker run --network none --rm --gpus '"device=0"' --ipc=host \
    -v "<Input>/:/input:ro" -v "<Output>:/output" <image> \
    /usr/local/bin/Process_SkillEval.sh <task>  # <task> in {GRS, OSATS, TRACK}
  ```
- 単一タスク対応は `<task>` 引数無しで実行し、直接対象タスクを実行。
- それぞれ上記「入出力仕様」に従ったCSV/TXTを `/output` に生成。

## 8. 提出・Write-up
- Docker命名: `oss25_<team_name>:v<version>`（例: v1, v2, v3）。
- Synapse に Docker と Write-up を提出（Certified User 必要）。
- 最終提出後、プロジェクトを公開共有（Challenge Admin に共有）。
- Write-up はテンプレートに従い、締切は 2025-09-15 23:59 GMT。

## 9. 実装メモ（注意点）
- Task 3 の Val 構造は Train と異なるため、ローダ実装時に個別対応。
- 手のKP点数・`xyc/xyv` の差異は原文に揺れがあるため、配布評価スクリプトの仕様をソースで確認して整合させる。
- Task 1/2 のメトリクス（F1, Expected Cost）実装は公式スクリプト準拠で検証。
- 出力CSVはヘッダの有無や列名の厳密性に注意（評価スクリプト前提に合わせる）。

---
出典: `OSS2025_pub.md`（コンペ公式サイト記載の要件を転載）
