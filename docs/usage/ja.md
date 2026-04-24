# MAFIS 利用ガイド (日本語)

新規インストールから日常運用までの手順書です。概要は
[メイン README](../../README.md)、設計思想は
[design-v2.2.md](../../design-v2.2.md)、Phase 1 の正式評価は
[docs/MVP_EVALUATION.md](../MVP_EVALUATION.md) を参照してください。

---

## 1. セットアップ (約10分)

### 1.1 必要条件

- Python 3.13 以上
- [Ollama](https://ollama.com/) (ローカル LLM ランタイム)
- 無料 API キー 4種 (下記 1.3)

### 1.2 Ollama + モデル

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# クルーが使用する 16k コンテキストモデル
ollama pull qwen2.5:7b-16k
ollama pull llama3.1:8b-16k
```

### 1.3 API キー発行

| サービス | 用途 | リンク |
|---|---|---|
| **Finnhub** | 米国株ファンダ・株価 | [finnhub.io](https://finnhub.io/) |
| **FRED** | マクロ経済指標 (Economist) | [fredaccount.stlouisfed.org](https://fredaccount.stlouisfed.org/apikeys) |
| **OpenDART** | 韓国株ファンダ | [opendart.fss.or.kr](https://opendart.fss.or.kr/mngInfo/mngInfoMain.do) |
| **Telegram** (任意) | プッシュ通知 | `@BotFather` で `/newbot` |

`.env.example` を `.env` にコピーしてキーを記入:

```bash
cp .env.example .env
# .env を編集
FINNHUB_API_KEY=...
FRED_API_KEY=...
DART_API_KEY=...            # 韓国株分析時のみ必須
TELEGRAM_BOT_TOKEN=...      # 任意
TELEGRAM_CHAT_ID=...        # 任意
```

### 1.4 Python 環境

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 1.5 動作確認

```bash
python scripts/verify_env.py    # API キーと Ollama への到達可否を確認
pytest                          # "480 passed" が出れば正常
```

---

## 2. 初回実行 — NVDA エンドツーエンド

### 2.1 銘柄登録 (バリューチェーン草稿を自動生成)

```bash
python scripts/onboard_ticker.py NVDA --tier 1 --notes "初ターゲット"
```

約3〜5分。実施内容:

1. Finnhub から会社名・業種・同業他社リスト取得
2. SEC EDGAR から最新 10-K をダウンロードし ChromaDB にインデックス
3. 地政学ニュース (GDELT + Google News) を取得
4. Qwen 2.5 7B が8セクションのバリューチェーン・ブリーフ草稿を作成
5. `docs/value_chains/NVDA.draft.md` に保存
6. `config/tickers.yaml` に Tier 1 として登録

### 2.2 草稿レビュー (人手作業、2〜3分)

`docs/value_chains/NVDA.draft.md` を開いて:

- **Vulnerable links** セクションを重点確認 (Skeptic エージェントが最も参照する箇所)
- `[?UNCERTAIN]` 付きの項目は LLM が自ら低信頼を宣言したもの。検証または削除
- 問題なければ `.draft.md` を `.md` にリネームして有効化:

```bash
mv docs/value_chains/NVDA.draft.md docs/value_chains/NVDA.md
```

### 2.3 クルー実行

```bash
python scripts/run_crew.py NVDA
```

約15〜20分 (6エージェント逐次)。出力:

- `reports/NVDA_YYYYMMDD_HHMM.crew.md` — 6セクション + 監査ブロック
- `reports/NVDA_YYYYMMDD_HHMM.crew.meta.txt` — 各エージェントの実行時間・モデル
- `data/portfolio.sqlite` の `paper_trades` テーブルに Steward 判定を自動記録
- Telegram 設定済みなら韓国語要約を自動プッシュ

### 2.4 韓国株も同じフロー

```bash
python scripts/onboard_ticker.py 005930 --tier 1   # サムスン電子
python scripts/run_crew.py 005930
```

DART ディスパッチャが6桁の KRX コードを自動検知して OpenDART 経由でファンダを取得。

---

## 3. レポートの読み方

### 3.1 6セクション構成

| Part | エージェント | 役割 |
|---|---|---|
| 1 | Economist | FFレート・CPI・為替などマクロ背景 |
| 2 | Analyst | 事業要約・財務健全性の7セクション |
| 3 | Valuer | PER / EV/EBITDA / リバース DCF による市場内含成長率 |
| 4 | Skeptic | Bull シーシスへの5つの反論 (Llama — 別モデル使用) |
| 5 | Defender | Skeptic 反論への回答: DEFENDED / CONCEDED |
| 6 | Steward | 最終判定 — BUY / HOLD / PASS + Conviction 1〜5 |

### 3.2 System Audit ブロック (レポート末尾)

Steward の直後に自動付与。例:

```
### System Audit — Discipline Matrix Enforcement
- Raw Steward labels: NEUTRALIZED=1, SURVIVED=1.
- Defender labels (authoritative): DEFENDED=1, CONCEDED=4.
- Steward mis-translated Defender labels.
- Effective labels: NEUTRALIZED=1, SURVIVED=4.
- Reported Verdict: BUY / Conviction: 4.
- Matrix ceiling: PASS / Conviction 1.
- VIOLATION: reported Verdict=BUY exceeds matrix ceiling PASS.
```

**読み方**:
- Steward LLM は BUY C4 と判定したが、Defender は5つの反論中1つしか防御できず4つ譲歩
- Python 監査が自動的に判定を下げ、実際の最終判定は **PASS Conviction 1**
- Steward のナラティブは原文のまま保存。下流 (paper ledger, Telegram 要約) は監査修正値を使用

### 3.3 Citation Grounding ブロック

`[Source: edgar.*]` 引用の数値が実際に 10-K に存在するかを自動検証:

```
## System Audit — Citation Grounding
1 ungrounded claim(s):
- Claim '15-20%' in section 'mdna_highlights' not found in retrieved passages.
```

意味: LLM がもっともらしい引用を付けたが、その数値は実際の 10-K に無い → 信頼しないこと。

### 3.4 判定の意味

- **BUY C5**: 最高確信度。Defender が5つ全て DEFENDED。ポジション 5〜8%
- **BUY C3〜4**: Bull 優勢だが一部譲歩あり。ポジション 2〜5%
- **HOLD C2**: 拮抗。既存ポジション維持、新規追加なし
- **HOLD C1**: 微妙。実質 PASS に近い
- **PASS C1**: Bear 優勢。ポジションなし

---

## 4. 日常運用

### 4.1 ポートフォリオ管理

```bash
# ポジション追加
python scripts/portfolio_cli.py add NVDA --shares 10 --cost 5000 --tier 1

# 一覧
python scripts/portfolio_cli.py list

# 現値基準の比重 (Finnhub ライブ)
python scripts/portfolio_cli.py weights

# Steward 推奨比重 vs 現在比重
python scripts/portfolio_cli.py gap NVDA --low 3 --high 5
# 出力例: "Already at 4.2% (suggestion 3.0-5.0% — within band, no action)"
```

### 4.2 ペーパートレード収益率追跡

クルー実行時に paper_trades へ自動記録されます。数日〜数週間後:

```bash
python scripts/paper_ledger.py list                 # 全記録
python scripts/paper_ledger.py returns              # 現値基準の損益
python scripts/paper_ledger.py summary              # 勝率・verdict 別平均収益
```

`summary` の例:

```
By verdict
  BUY:  n=5  avg=+4.20%  win rate=60.0%
  HOLD: n=2  avg=+1.50%
  PASS: n=3  avg=-2.80%

By conviction
  C4: n=3  avg=+6.10%
  C2: n=2  avg=-0.30%

Audit effect (original BUY verdicts)
  BUYs that cleared audit: +7.50%
  BUYs downgraded by audit: -3.20%
```

最後のブロックが最も重要: **Python 監査が下げた BUY が実際にパフォーマンス劣後していれば**、監査システムがアルファを生んでいる客観的証拠となる。

### 4.3 Chain Alerts (ニュース → ポジション連鎖通知)

バリューチェーングラフ上のノード (TSMC / ASML / Siemens など) がニュースに登場すると、関連 Tier 1 銘柄に通知:

```bash
# 手動実行
python scripts/scan_chain_alerts.py --dedup --hops 2

# 立会時間の毎時 cron
0 9-16 * * 1-5  cd ~/MAFIS && /path/to/.venv/bin/python \
    scripts/scan_chain_alerts.py --dedup --telegram \
    >> /var/log/mafis_alerts.log 2>&1
```

`--dedup`: 48時間以内の同じ通知は再送しない (SQLite ledger 利用)。

### 4.4 Tier 3 昇格推奨 (プリフィルタ)

Tier 3 (登録のみで処理されない銘柄) の中で最近ニュースに頻出する銘柄を発見:

```bash
python scripts/prefilter_scan.py --graph-context --semantic
```

- Stage 1: キーワードマッチ
- Stage 2: バリューチェーン・コンテキスト (Tier 1/2 企業の供給元・同業言及)
- Stage 3: Qwen によるマテリアリティ判定

### 4.5 プロンプト調整後の回帰防止

プロンプト変更 → 同銘柄再実行 → 前回レポートと構造品質を比較:

```bash
python scripts/regression_compare.py \
    reports/NVDA_20260424_1715.crew.md \
    reports/NVDA_20260425_0900.crew.md
```

citation_rate / refusal_count / 監査違反数 / edgar 引用数などを IMPROVED / REGRESSED / NEUTRAL で自動分類。

---

## 5. よくあるエラー

### 5.1 `FINNHUB_API_KEY not set`

`.env` がプロジェクトルートにあるか、キー前後の空白がないか確認。

### 5.2 SEC EDGAR 403 Forbidden

SEC フェアユース・ポリシーは User-Agent にメールアドレスを要求。既定値は
`MAFIS research ccy5123ccy@gmail.com`。`src/wise_investor/rag/edgar.py` 内を自分のメールに変更推奨:

```python
USER_AGENT = "YourName research your@email.com"
```

### 5.3 DART レスポンスに `status: 013`

多くは `corp_code` 不一致または該当年度の公示なし。デバッグ:

```bash
python scripts/probe_dart.py 005930 --year 2024 --dump
```

### 5.4 クルー実行が遅すぎる (> 20分)

- `ollama ps` で GPU / CPU 使用状況を確認
- Ollama のメモリ不足の可能性 — `ollama stop` 後に再起動
- Skeptic (Llama) ↔ Defender/Steward (Qwen) のモデルスワップで 30秒〜2分かかるのは正常

### 5.5 Chain alerts に何も出ない

```bash
# バリューチェーングラフを再構築
python scripts/build_value_chain_graph.py

# 個別銘柄ニュースを手動確認
python scripts/probe_geopolitics.py NVDA --timespan 3days
```

---

## 6. ディレクトリ構成

```
MAFIS/
├── src/wise_investor/         # メインパッケージ
│   ├── agents/                # 6エージェント
│   ├── data/                  # Finnhub / DART / FRED クライアント
│   ├── rag/                   # SEC EDGAR 10-K ChromaDB
│   ├── geopolitics/           # GDELT + Google News
│   ├── alerts/                # chain alerts + dedup ledger
│   ├── filters/               # プリフィルタ 3段階
│   ├── onboarding/            # 銘柄追加自動化
│   ├── portfolio/             # ポジション SQLite
│   ├── paper_trading/         # Steward 判定の時系列追跡
│   └── regression/            # レポート差分ツール
├── scripts/                   # CLI エントリポイント
├── docs/value_chains/         # バリューチェーン・ブリーフ
├── config/tickers.yaml        # 3-Tier レジストリ
└── data/                      # portfolio.sqlite, chroma/, edgar_cache/, facts_cache/
```

---

## 7. 核心原則

- **ローカル優先、API 最後**: 外部 LLM API コストゼロ。Finnhub / FRED /
  GDELT / DART はすべて無料公開 API。
- **LLM は判断、Python は計算**: すべての数値は Python ツールが算出。
  LLM はナラティブのみ。
- **再現性**: `temperature=0`, `seed=42` → 同じ facts cache なら
  エージェント出力はバイト単位で一致。
- **多層監査**: discipline matrix + speculative-language 検知 +
  Defender-aware 補正 + citation grounding + Skeptic mandate 遵守。
- **ペーパートレード優先**: 実売買の前に数週間〜数ヶ月
  `paper_ledger.py summary` で実績を検証。

---

## 8. 追加資料

- [設計ドキュメント](../../design-v2.2.md)
- [MVP 正式評価](../MVP_EVALUATION.md)
- [GitHub repo](https://github.com/ccy5123/mafis)

問題発生時は GitHub の Issue へ。
