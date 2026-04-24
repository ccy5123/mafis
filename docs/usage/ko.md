# MAFIS 사용 설명서 (한국어)

Wise Investor System(MAFIS)을 **처음부터 일상 운영까지** 안내합니다.
개요가 필요하면 [메인 README](../../README.md)를, 설계 의도는
[design-v2.2.md](../../design-v2.2.md), MVP 평가는
[docs/MVP_EVALUATION.md](../MVP_EVALUATION.md)를 참고하세요.

---

## 1. 빠른 설치 (약 10분)

### 1.1 사전 요구사항

- Python 3.13 이상
- [Ollama](https://ollama.com/) (로컬 LLM 런타임)
- 무료 API 키 4종 (아래 1.3 참고)

### 1.2 Ollama + 모델

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# 크루에서 사용하는 16k 컨텍스트 모델 다운로드
ollama pull qwen2.5:7b-16k
ollama pull llama3.1:8b-16k
```

### 1.3 API 키 발급

| 서비스 | 용도 | 링크 |
|---|---|---|
| **Finnhub** | 미국 종목 재무/시세 | [finnhub.io](https://finnhub.io/) |
| **FRED** | 거시경제 지표 (Economist) | [fredaccount.stlouisfed.org](https://fredaccount.stlouisfed.org/apikeys) |
| **OpenDART** | 한국 종목 재무 | [opendart.fss.or.kr](https://opendart.fss.or.kr/mngInfo/mngInfoMain.do) |
| **Telegram** (선택) | 푸시 알림 | `@BotFather`에서 `/newbot` |

`.env.example`을 `.env`로 복사 후 위 키 값 입력:

```bash
cp .env.example .env
# .env 편집
FINNHUB_API_KEY=...
FRED_API_KEY=...
DART_API_KEY=...            # 한국 종목 분석 시 필수
TELEGRAM_BOT_TOKEN=...      # 선택
TELEGRAM_CHAT_ID=...        # 선택
```

### 1.4 Python 환경

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 1.5 환경 검증

```bash
python scripts/verify_env.py    # 모든 API 키 + Ollama 도달 가능 확인
pytest                          # 480 passed 출력되어야 정상
```

---

## 2. 첫 실행 — NVDA 전체 파이프라인

### 2.1 종목 등록 (자동 밸류체인 초안 생성)

```bash
python scripts/onboard_ticker.py NVDA --tier 1 --notes "첫 타깃"
```

약 3~5분 소요. 이 명령어가 수행하는 것:

1. Finnhub에서 회사명 / 업종 / peer 목록 수집
2. SEC EDGAR에서 최신 10-K 다운로드 + ChromaDB 인덱싱
3. 최근 지정학 뉴스 (GDELT + Google News) 수집
4. Qwen 2.5 7B가 8개 섹션 밸류체인 브리프 초안 작성
5. `docs/value_chains/NVDA.draft.md` 저장
6. `config/tickers.yaml`에 Tier 1로 등록

### 2.2 초안 검토 (사람 작업, 2~3분)

`docs/value_chains/NVDA.draft.md` 열어서:

- **Vulnerable links** 섹션을 중점 검토 (Skeptic agent가 가장 많이 쓰는 자료)
- `[?UNCERTAIN]` 표시된 항목: LLM이 자체 신뢰도 낮음을 표시. 검증 또는 삭제
- 이상 없으면 `.draft.md` → `.md`로 이름 변경:

```bash
mv docs/value_chains/NVDA.draft.md docs/value_chains/NVDA.md
```

### 2.3 크루 실행

```bash
python scripts/run_crew.py NVDA
```

약 15~20분 소요 (6개 agent 순차 실행). 출력:

- `reports/NVDA_YYYYMMDD_HHMM.crew.md` — 6개 섹션 + 감사 블록
- `reports/NVDA_YYYYMMDD_HHMM.crew.meta.txt` — 각 agent 실행 시간 / 모델
- `data/portfolio.sqlite`의 `paper_trades` 테이블에 Steward 판정 자동 기록
- Telegram 설정 시 한국어 요약 자동 푸시

### 2.4 한국 종목도 동일한 흐름

```bash
python scripts/onboard_ticker.py 005930 --tier 1   # 삼성전자
python scripts/run_crew.py 005930
```

DART dispatcher가 자동으로 감지해서 한국 재무 데이터를 주입합니다.

---

## 3. 리포트 읽는 법

### 3.1 6개 섹션 구조

| Part | Agent | 역할 |
|---|---|---|
| 1 | Economist | Fed 금리 / CPI / 환율 등 매크로 배경 |
| 2 | Analyst | 비즈니스 요약, 재무 건전성, 7개 섹션 |
| 3 | Valuer | PER / EV/EBITDA / 역방향 DCF로 시장 내재 성장률 |
| 4 | Skeptic | 5개 반박 (Bull thesis 공격), Llama 다른 모델 사용 |
| 5 | Defender | Skeptic 반박 5개에 대해 DEFENDED / CONCEDED 답변 |
| 6 | Steward | 최종 판정 — BUY / HOLD / PASS + Conviction 1~5 |

### 3.2 System Audit 블록 (리포트 끝)

Steward 바로 아래 자동 첨부됩니다. 예시:

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

**읽는 법**:
- Steward LLM이 BUY C4 발행했지만, Defender는 1 DEFENDED + 4 CONCEDED (5개 중 4개 패배)
- Python 감사가 자동 downgrade → 실제 최종 판정은 **PASS / Conviction 1**
- Steward 내러티브는 그대로 보존되지만 다운스트림(paper ledger, Telegram 요약)은 감사 수정값 사용

### 3.3 Citation Grounding 블록

`[Source: edgar.*]` 인용이 있는 문장에서 인용한 숫자가 실제 10-K passage에 존재하는지 자동 검증. 문제 있으면 다음처럼 표시:

```
## System Audit — Citation Grounding
1 ungrounded claim(s):
- Claim '15-20%' in section 'mdna_highlights' not found in retrieved passages.
```

의미: LLM이 그럴듯한 인용을 붙였지만 실제 10-K에는 그 숫자가 없음 → 신뢰하지 말 것.

### 3.4 판정 체계

- **BUY Conviction 5**: 최고 확신. Defender 5개 모두 DEFENDED. 포지션 5~8%.
- **BUY Conviction 3~4**: Bull 다수 방어 + 일부 인정. 포지션 2~5%.
- **HOLD Conviction 2**: Bull/Bear 비슷. 기존 포지션 유지, 신규 진입 보류.
- **HOLD Conviction 1**: 균형 아슬아슬. PASS 가깝다고 해석.
- **PASS Conviction 1**: Bear 다수 승리. 진입 안 함.

---

## 4. 일상 운영

### 4.1 포트폴리오 관리

```bash
# 포지션 추가
python scripts/portfolio_cli.py add NVDA --shares 10 --cost 5000 --tier 1

# 전체 목록
python scripts/portfolio_cli.py list

# 현재 시가 기준 비중 (Finnhub 라이브)
python scripts/portfolio_cli.py weights

# Steward 제안 비중 vs 현재 비중 비교
python scripts/portfolio_cli.py gap NVDA --low 3 --high 5
# 출력 예: "Already at 4.2% (suggestion 3.0-5.0% — within band, no action)"
```

### 4.2 페이퍼 트레이딩 수익률 추적

크루 실행 시 자동으로 paper_trades에 기록됨. 며칠~몇 주 뒤:

```bash
python scripts/paper_ledger.py list                 # 전체 기록
python scripts/paper_ledger.py returns              # 현재 시가 기준 손익
python scripts/paper_ledger.py summary              # 승률, verdict별 평균 수익률
```

`summary` 출력 예:

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

마지막 부분이 특히 중요합니다: **Python 감사가 downgrade한 BUY들이 실제로 부진했다면**, 감사 시스템이 알파를 기여했다는 객관적 증거입니다.

### 4.3 Chain Alerts (뉴스 → 포지션 연쇄 알림)

value chain 그래프에 있는 노드(예: TSMC, ASML, Siemens)가 뉴스에 등장하면 연결된 Tier 1 종목에 알림:

```bash
# 수동 실행
python scripts/scan_chain_alerts.py --dedup --hops 2

# cron (장중 매시간)
0 9-16 * * 1-5  cd ~/MAFIS && /path/to/.venv/bin/python \
    scripts/scan_chain_alerts.py --dedup --telegram \
    >> /var/log/mafis_alerts.log 2>&1
```

`--dedup`: 48시간 이내 같은 알림은 재전송 안 함 (SQLite ledger 활용).

### 4.4 Tier 3 승격 추천 (사전 필터)

Tier 3 (등록만 하고 처리 안 하는 종목)에서 최근 뉴스에 자주 언급된 종목을 찾아 Tier 2 승격 추천:

```bash
python scripts/prefilter_scan.py --graph-context --semantic
```

- Stage 1: 키워드 매칭
- Stage 2: value chain 그래프 컨텍스트 (Tier 1/2 기업들의 공급사/peer 언급)
- Stage 3: Qwen이 "이 헤드라인이 실제로 투자 판단에 material한가?"로 필터링

### 4.5 프롬프트 튜닝 후 회귀 방지

프롬프트 조정 → 같은 종목 재실행 → 이전 리포트와 구조적 품질 비교:

```bash
python scripts/regression_compare.py \
    reports/NVDA_20260424_1715.crew.md \
    reports/NVDA_20260425_0900.crew.md
```

citation_rate / refusal_count / 감사 위반 개수 / edgar 인용 개수 등이 자동으로 IMPROVED / REGRESSED / NEUTRAL로 분류됩니다.

---

## 5. 자주 발생하는 오류

### 5.1 `FINNHUB_API_KEY not set`

`.env` 파일이 프로젝트 루트에 있는지, 키 앞뒤 공백이 없는지 확인.

### 5.2 SEC EDGAR 403 Forbidden

SEC의 fair-use 정책은 User-Agent에 이메일 주소를 요구합니다. `src/wise_investor/rag/edgar.py`의 `USER_AGENT` 기본값은 `MAFIS research ccy5123ccy@gmail.com`인데, 본인 이메일로 수정하는 것을 권장:

```python
USER_AGENT = "YourName research your@email.com"
```

### 5.3 DART 응답에 `status: 013`

대부분 `corp_code` 불일치 또는 해당 연도 공시 없음. 다음 명령으로 디버깅:

```bash
python scripts/probe_dart.py 005930 --year 2024 --dump
```

### 5.4 크루 실행이 너무 느림 (> 20분)

- `ollama ps`로 GPU/CPU 사용 확인
- Ollama 메모리 부족 가능성 — `ollama stop` 후 재실행
- Skeptic (Llama) ↔ Defender/Steward (Qwen) 모델 swap 오버헤드 30초~2분 정상

### 5.5 Chain alerts가 아무것도 안 나옴

```bash
# value chain 그래프 재구축
python scripts/build_value_chain_graph.py

# 개별 종목 뉴스 수동 확인
python scripts/probe_geopolitics.py NVDA --timespan 3days
```

---

## 6. 디렉토리 구조 요약

```
MAFIS/
├── src/wise_investor/         # 메인 패키지
│   ├── agents/                # 6개 agent (Economist/Analyst/Valuer/Skeptic/Defender/Steward)
│   ├── data/                  # Finnhub / DART / FRED 클라이언트
│   ├── rag/                   # SEC EDGAR 10-K ChromaDB
│   ├── geopolitics/           # GDELT + Google News
│   ├── alerts/                # Chain alerts + dedup ledger
│   ├── filters/               # Pre-filter 3 stages
│   ├── onboarding/            # 종목 추가 자동화
│   ├── portfolio/             # 포지션 SQLite
│   ├── paper_trading/         # Steward verdict 시계열 추적
│   └── regression/            # 리포트 diff 도구
├── scripts/                   # CLI 진입점
├── docs/value_chains/         # 수동/자동 밸류체인 브리프
├── config/tickers.yaml        # 3-Tier 종목 레지스트리
└── data/                      # portfolio.sqlite, chroma/, edgar_cache/, facts_cache/
```

---

## 7. 핵심 원칙 요약

- **로컬 우선, API 최후**: 외부 LLM API 비용 0원. Finnhub/FRED/GDELT/DART 모두 무료 공개 API.
- **LLM은 판단, Python은 계산**: 모든 수치는 Python 도구가 계산. LLM은 내러티브만.
- **재현성**: `temperature=0`, `seed=42` → 같은 facts cache면 agent 출력 byte-identical.
- **다층 감사**: discipline matrix + speculative-language + Defender-aware + citation grounding + Skeptic mandate.
- **페이퍼 트레이딩 먼저**: 실제 매매 전 몇 주~몇 달 `paper_ledger.py summary`로 실제 수익률 검증.

---

## 8. 추가 자료

- [설계 문서](../../design-v2.2.md) — 전체 아키텍처
- [MVP 평가 문서](../MVP_EVALUATION.md) — Phase 1 4대 질문 공식 답변
- [GitHub repo](https://github.com/ccy5123/mafis)

문제 발생 시 이슈 등록해주세요.
