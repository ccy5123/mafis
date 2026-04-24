# MAFIS 使用指南 (简体中文)

从全新安装到日常运营的完整手册。总览请参见
[主 README](../../README.md); 架构思想参见
[design-v2.2.md](../../design-v2.2.md); Phase 1 正式评估参见
[docs/MVP_EVALUATION.md](../MVP_EVALUATION.md)。

---

## 1. 安装 (约 10 分钟)

### 1.1 前置条件

- Python 3.13 或以上
- [Ollama](https://ollama.com/) (本地 LLM 运行时)
- 4 个免费 API key (见 1.3)

### 1.2 Ollama 与模型

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# Crew 使用的 16k 上下文模型
ollama pull qwen2.5:7b-16k
ollama pull llama3.1:8b-16k
```

### 1.3 API key 申请

| 服务 | 用途 | 链接 |
|---|---|---|
| **Finnhub** | 美股基本面/行情 | [finnhub.io](https://finnhub.io/) |
| **FRED** | 宏观指标 (Economist 使用) | [fredaccount.stlouisfed.org](https://fredaccount.stlouisfed.org/apikeys) |
| **OpenDART** | 韩股基本面 | [opendart.fss.or.kr](https://opendart.fss.or.kr/mngInfo/mngInfoMain.do) |
| **Telegram** (可选) | 推送通知 | `@BotFather` 发送 `/newbot` |

复制 `.env.example` 为 `.env` 并填入 key:

```bash
cp .env.example .env
# 编辑 .env
FINNHUB_API_KEY=...
FRED_API_KEY=...
DART_API_KEY=...            # 分析韩股时必须
TELEGRAM_BOT_TOKEN=...      # 可选
TELEGRAM_CHAT_ID=...        # 可选
```

### 1.4 Python 环境

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 1.5 环境验证

```bash
python scripts/verify_env.py    # 检查 API key 及 Ollama 可达性
pytest                          # 应输出 "480 passed"
```

---

## 2. 首次运行 — NVDA 端到端

### 2.1 添加 ticker (自动生成价值链初稿)

```bash
python scripts/onboard_ticker.py NVDA --tier 1 --notes "首个目标"
```

约 3~5 分钟。执行内容:

1. 从 Finnhub 获取公司名/行业/同业列表
2. 从 SEC EDGAR 下载最新 10-K 并索引到 ChromaDB
3. 抓取地缘政治新闻 (GDELT + Google News)
4. Qwen 2.5 7B 起草 8 节价值链简报
5. 保存至 `docs/value_chains/NVDA.draft.md`
6. 在 `config/tickers.yaml` 中登记为 Tier 1

### 2.2 审阅初稿 (人工步骤, 2~3 分钟)

打开 `docs/value_chains/NVDA.draft.md`:

- 重点检查 **Vulnerable links** 部分 (Skeptic agent 主要引用来源)
- `[?UNCERTAIN]` 标记的条目是 LLM 自我声明低置信度 — 确认或删除
- 审阅无误后将 `.draft.md` 重命名为 `.md` 启用:

```bash
mv docs/value_chains/NVDA.draft.md docs/value_chains/NVDA.md
```

### 2.3 运行 crew

```bash
python scripts/run_crew.py NVDA
```

约 15~20 分钟 (6 个 agent 顺序执行)。产出:

- `reports/NVDA_YYYYMMDD_HHMM.crew.md` — 6 节报告 + 审计块
- `reports/NVDA_YYYYMMDD_HHMM.crew.meta.txt` — 每个 agent 耗时与模型信息
- `data/portfolio.sqlite` 的 `paper_trades` 表自动插入 Steward 判定记录
- 若配置了 Telegram 则自动推送韩语摘要

### 2.4 韩股同样流程

```bash
python scripts/onboard_ticker.py 005930 --tier 1   # 三星电子
python scripts/run_crew.py 005930
```

DART 分发器会自动识别 6 位 KRX 代码并通过 OpenDART 获取数据。

---

## 3. 如何阅读报告

### 3.1 6 节结构

| Part | Agent | 职责 |
|---|---|---|
| 1 | Economist | 联邦基金利率、CPI、汇率等宏观背景 |
| 2 | Analyst | 7 节业务摘要 + 财务健康度报告 |
| 3 | Valuer | PER / EV-EBITDA / 反向 DCF 算出市场隐含成长率 |
| 4 | Skeptic | 对 Bull thesis 的 5 条反驳 (Llama — 异构模型) |
| 5 | Defender | 针对 Skeptic 5 条反驳: DEFENDED 或 CONCEDED |
| 6 | Steward | 最终判定 — BUY / HOLD / PASS + Conviction 1-5 |

### 3.2 System Audit 块 (报告末尾)

附加在 Steward 之后。示例:

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

**解读方式**:
- Steward LLM 发出了 BUY C4, 但 Defender 在 5 条反驳中只防御住 1 条, 让步 4 条
- Python 审计自动下调判定, 实际最终判定为 **PASS Conviction 1**
- Steward 叙述原样保留; 下游 (paper ledger, Telegram 摘要) 使用审计修正值

### 3.3 Citation Grounding 块

检查每个 `[Source: edgar.*]` 引用的数字是否真的出现在 10-K 段落中:

```
## System Audit — Citation Grounding
1 ungrounded claim(s):
- Claim '15-20%' in section 'mdna_highlights' not found in retrieved passages.
```

含义: LLM 附上了看似可信的引用, 但这个数字在实际 10-K 里并不存在 — 视为不可信。

### 3.4 判定语义

- **BUY C5**: 最高置信度, Defender 5 条全部 DEFENDED。仓位 5~8%
- **BUY C3~4**: Bull 占多数但有部分让步。仓位 2~5%
- **HOLD C2**: 势均力敌。保持既有持仓, 不新增
- **HOLD C1**: 勉强 HOLD, 实质接近 PASS
- **PASS C1**: Bear 占多数。不建仓

---

## 4. 日常运营

### 4.1 组合管理

```bash
# 添加持仓
python scripts/portfolio_cli.py add NVDA --shares 10 --cost 5000 --tier 1

# 查看全部
python scripts/portfolio_cli.py list

# 现价比重 (Finnhub 实时)
python scripts/portfolio_cli.py weights

# Steward 建议比重 vs 当前比重
python scripts/portfolio_cli.py gap NVDA --low 3 --high 5
# 输出示例: "Already at 4.2% (suggestion 3.0-5.0% — within band, no action)"
```

### 4.2 纸面交易绩效追踪

Crew 运行时自动写入 paper_trades 表。数日或数周后:

```bash
python scripts/paper_ledger.py list                 # 全部记录
python scripts/paper_ledger.py returns              # 现价损益
python scripts/paper_ledger.py summary              # 胜率、各 verdict 平均收益
```

`summary` 示例:

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

最后一个板块最关键: **如果被审计下调的 BUY 实际收益劣于未被下调的 BUY**, 就是 Python 审计层在贡献 alpha 的客观证据。

### 4.3 Chain alerts (新闻 → 持仓联动通知)

当价值链图中的节点 (TSMC / ASML / Siemens 等) 出现在新闻时, N 跳内可达的 Tier 1 目标会收到通知:

```bash
# 单次扫描
python scripts/scan_chain_alerts.py --dedup --hops 2

# 盘中逐小时 cron
0 9-16 * * 1-5  cd ~/MAFIS && /path/to/.venv/bin/python \
    scripts/scan_chain_alerts.py --dedup --telegram \
    >> /var/log/mafis_alerts.log 2>&1
```

`--dedup`: 48 小时内同一 alert 不重复推送 (SQLite ledger)。

### 4.4 Tier 3 提升建议 (预过滤)

从 Tier 3 (仅登记、不处理) 中筛选出近期频繁被新闻提及的标的:

```bash
python scripts/prefilter_scan.py --graph-context --semantic
```

- Stage 1: 关键字匹配
- Stage 2: 价值链上下文 (Tier 1/2 企业的供应商/同业提及)
- Stage 3: Qwen 判断是否真正 material

### 4.5 提示/模型调整后的回归保护

修改提示或升级模型后, 对比旧报告与新报告的结构化质量:

```bash
python scripts/regression_compare.py \
    reports/NVDA_20260424_1715.crew.md \
    reports/NVDA_20260425_0900.crew.md
```

citation_rate / refusal_count / 审计违规数 / edgar 引用数等自动归类为 IMPROVED / REGRESSED / NEUTRAL。

---

## 5. 常见错误

### 5.1 `FINNHUB_API_KEY not set`

确认 `.env` 位于项目根目录, key 前后无空格。

### 5.2 SEC EDGAR 返回 403

SEC 公平使用政策要求 User-Agent 含邮箱地址。默认值为
`MAFIS research ccy5123ccy@gmail.com`。建议修改 `src/wise_investor/rag/edgar.py` 为自己的邮箱:

```python
USER_AGENT = "YourName research your@email.com"
```

### 5.3 DART 响应 `status: 013`

多为 `corp_code` 不符或该年度无公告。调试:

```bash
python scripts/probe_dart.py 005930 --year 2024 --dump
```

### 5.4 Crew 运行过慢 (> 20 分钟)

- 用 `ollama ps` 查看 GPU / CPU 负载
- Ollama 可能内存不足 — `ollama stop` 后重启
- Skeptic (Llama) ↔ Defender/Steward (Qwen) 模型切换耗时 30 秒~2 分钟属正常

### 5.5 Chain alerts 为空

```bash
# 重新构建价值链图
python scripts/build_value_chain_graph.py

# 手动检测某个标的
python scripts/probe_geopolitics.py NVDA --timespan 3days
```

---

## 6. 目录结构

```
MAFIS/
├── src/wise_investor/         # 主包
│   ├── agents/                # 6 个 crew agents
│   ├── data/                  # Finnhub / DART / FRED 客户端
│   ├── rag/                   # SEC EDGAR 10-K ChromaDB
│   ├── geopolitics/           # GDELT + Google News
│   ├── alerts/                # chain alerts + dedup ledger
│   ├── filters/               # pre-filter 3 阶段
│   ├── onboarding/            # 添加 ticker 自动化
│   ├── portfolio/             # 持仓 SQLite
│   ├── paper_trading/         # Steward 判定时间序列追踪
│   └── regression/            # 报告差异工具
├── scripts/                   # CLI 入口
├── docs/value_chains/         # 手写/自动草稿
├── config/tickers.yaml        # 3-Tier 注册表
└── data/                      # portfolio.sqlite, chroma/, edgar_cache/, facts_cache/
```

---

## 7. 核心原则

- **本地优先, API 最后**: 外部 LLM API 成本为零。Finnhub / FRED / GDELT / DART 均为免费公开 API。
- **LLM 做判断, Python 做计算**: 报告中每个数字都来自 Python 工具。LLM 仅生成叙述。
- **可重现**: `temperature=0`, `seed=42` → 相同 facts cache 下 agent 输出逐字节一致。
- **多层审计**: discipline matrix + 投机语言检测 + Defender-aware 校正 + citation grounding + Skeptic mandate 合规。
- **先纸面交易再实盘**: 真金投入前先用 `paper_ledger.py summary` 积累数周数据, 验证 BUY 是否真能跑赢 PASS。

---

## 8. 更多资源

- [设计文档](../../design-v2.2.md)
- [MVP 正式评估](../MVP_EVALUATION.md)
- [GitHub repo](https://github.com/ccy5123/mafis)

出现问题请在 GitHub 提 issue。
