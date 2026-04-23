# Wise Investor System

장기 펀더멘털 투자 분석을 위한 로컬-우선 멀티 에이전트 시스템.

설계 문서: [design-v2.2.md](design-v2.2.md)

## 현재 단계: Phase 0 — 스캐폴드 + 환경 검증

## 사전 요구사항

- Python 3.12 이상
- Ollama (로컬 LLM 실행) — [ollama.com](https://ollama.com/)
- FMP API 키 (무료) — [Financial Modeling Prep](https://site.financialmodelingprep.com/)

## 셋업

### 1. Ollama 설치 및 모델 다운로드

```bash
# WSL Ubuntu
curl -fsSL https://ollama.com/install.sh | sh

# 서비스 시작
ollama serve &

# 모델 pull (Phase 1에서 사용)
ollama pull llama3.1:8b      # Analyst, Valuer
ollama pull qwen2.5:7b       # Skeptic (로컬 모델 다양성)
```

### 2. FMP API 키 발급

1. [site.financialmodelingprep.com](https://site.financialmodelingprep.com/) 회원가입
2. 대시보드에서 API 키 복사
3. `.env.example`을 `.env`로 복사 후 `FMP_API_KEY`에 붙여넣기

### 3. Python 환경 구축 (uv 권장)

```bash
# uv 설치 (없을 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 의존성 설치
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

또는 pip 사용:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 4. 환경 검증

```bash
python scripts/verify_env.py
```

모든 항목이 OK로 나오면 Phase 1A로 진입.

## 디렉토리 구조

```
MAFIS/
├── design-v2.2.md          # 설계 문서
├── pyproject.toml
├── src/wise_investor/
│   ├── config.py           # 환경 설정 (재현성 포함)
│   ├── tools/              # Python 계산 도구 (Phase 1A)
│   ├── agents/             # CrewAI 에이전트 (Phase 1B)
│   ├── data/               # FMP/yfinance 클라이언트 (Phase 1A)
│   └── rag/                # ChromaDB 인터페이스 (Phase 1B+)
├── scripts/verify_env.py   # 환경 점검
├── tests/                  # pytest
└── docs/value_chains/      # 수동 밸류체인 Markdown
```

## Phase 1 로드맵

설계 문서 §10.2 참조. 현재 위치는 Phase 0 (스캐폴드).

- Phase 0: 프로젝트 스캐폴드 + 환경 검증 (1~2일)
- Phase 1A: Python 계산 도구 + 데이터 수집 (1주)
- Phase 1B: Analyst 에이전트 첫 실행 (1주)
- Phase 1C: Valuer + Skeptic + 순차 큐 (1~2주)
- Phase 1D: 품질 지표 + 보고서 + 4대 평가 (2~3주)

## 핵심 원칙

- **로컬 우선, API 최후**: Phase 1은 외부 LLM API 비용 0원
- **LLM은 판단, Python은 계산**: 수치는 `src/wise_investor/tools/`가, 해석은 LLM이
- **재현성**: `temperature=0`, `seed=42` 기본값

## Telegram 알림 (선택)

Tier 1 리포트가 완성되면 한국어 요약이 텔레그램으로 푸시됩니다.

1. 텔레그램 `@BotFather`에 `/newbot` → 이름/유저네임 설정 → 토큰 복사
2. 본인 봇에 아무 메시지나 한 번 보내기 (채팅 생성)
3. `https://api.telegram.org/bot<TOKEN>/getUpdates` 접속 → 응답에서 `chat.id` 복사
4. `.env`에 추가:
   ```
   TELEGRAM_BOT_TOKEN=<1단계 토큰>
   TELEGRAM_CHAT_ID=<3단계 chat id>
   ```
5. `scripts/run_crew.py <TICKER>` 또는 `scripts/daily_run.py` 실행 후 자동 알림 수신

설정하지 않으면 조용히 건너뜀 (에러 없음).
