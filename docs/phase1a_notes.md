# Phase 1A — 중요 참고 사항

Phase 1A 구축 중 발견된 FMP API 동작 특성 및 후속 단계(Phase 1B+)에서 프롬프트
설계자·Skeptic 전략에 영향을 주는 항목들을 기록합니다.

## 해결됨

### 1. FMP `/stable/quote`는 PE/EPS/sharesOutstanding 미반환

**원인:** 엔드포인트 스펙. 유료 티어 문제 아님. `/stable/quote`는 17개 키(price,
marketCap, volume, 일부 이동평균 등)만 반환.

**조치:** `cross_validate_quote`에서 `pe` 비교를 제거. PE 정보가 필요한 경우
`calculate_per()` 결과를 사용하고, 이를 yfinance의 `trailingPE`와 비교하는 별도
함수가 필요하면 Phase 2에서 추가.

**대안 소스:**
- `/stable/key-metrics` → `earningsYield` (= 1/PER)
- `/stable/ratios` → `priceToEarningsRatio`

### 2. FMP `key-metrics.evToEBITDA` 자동 alias 매칭 실패

**원인:** FMP가 acronym을 전부 대문자(`evToEBITDA`)로 emit. pydantic
`alias_generator=to_camel`은 `evToEbitda`로 변환하여 불일치.

**조치:** `KeyMetrics.ev_to_ebitda`에 `Field(alias="evToEBITDA")` 명시. 유사한
all-caps 필드가 다른 모델에 있는지는 Phase 1B 진입 전 빠르게 훑어볼 필요. 현재
`Ratios.enterprise_value_multiple`은 표준 camelCase라 문제 없음.

## 프롬프트 설계자에게 전달해야 할 운영 주의사항

### 3. `/stock-peers` 결과에 데이터 미비 종목 혼재 가능

**관찰:** AAPL peer 질의에 Nextpower(NXT) 같은 소형주가 포함되었고 재무제표 API
응답이 비어 있어 PER / EV/EBITDA가 None으로 반환됐습니다. `get_peer_multiples`
는 이를 row별 warning으로 graceful 처리하지만, **Analyst·Valuer 프롬프트는
결측 행을 비교 평균·중앙값 계산에서 제외해야 합니다.**

**권장 프롬프트 문구(초안):**
> "peer multiples 테이블에서 PER 또는 EV/EBITDA가 비어 있는 행은 데이터 미비로
> 간주하고 중앙값·평균값 계산, 백분위 산출, 언급에서 제외하라. 제외한 행은
> 보고서 말미의 'Data Gaps' 섹션에 symbol을 명시하라."

### 4. FMP reported multiples는 회계연도 종가 기준 → 현재가 기준 계산과 괴리

**관찰:** AAPL Phase 1A smoke에서 우리 계산 PER 35.68 vs FMP reported 34.09 (≈
4.66% 차이). 원인은 FMP의 `ratios.priceToEarningsRatio`가 FY25 회계연도 종가
기준이고, 우리 `calculate_per`는 현재가 기준. 주가 변동이 있으면 언제나 나타나는
구조적 차이.

**조치:** 현재 `diff_pct_vs_fmp`가 이 값을 그대로 보고함. 프롬프트에서 "5% 이하
차이는 시점 차이로 정상. 10% 이상이면 조사 대상"으로 해석하도록 안내.

### 5. 가장 최근 회계연도 보고서의 일부 메트릭이 None일 수 있음

**관찰:** FY25 KeyMetrics에 `evToEBITDA`는 alias 수정 후 정상 반환됨. 그러나
회계연도 직후에는 일부 FMP 파생 지표가 아직 채워지지 않은 상태일 수 있음.

**조치:** 모든 계산 결과의 `warnings` 필드를 보고서에 보존. 결측 필드는 가정하지
말고 명시하라는 원칙 유지.

## 도메인 데이터 한계 (Phase 3에서 본격 대응)

### 6. FMP 단일 지점 장애 (SPOF)

**내용:** Phase 1 데이터 파이프라인은 FMP 무료 티어에 강하게 의존. FMP 서비스
중단 또는 가격 인상 시 백업은 yfinance(스크래핑 불안정)뿐.

**Phase 3 대응:** SEC EDGAR XBRL 파싱을 최종 대체재(진실의 근원)로 승격. 설계
문서 v2.2 §3.2 "EDGAR의 전략적 위상 재정의" 참조.

### 7. EDGAR 수치 교차검증 Phase 1에서는 불가능

**내용:** v2.2에서 EDGAR XBRL 수치 파싱을 Phase 3로 이연. 따라서 Phase 1~2의
교차검증은 FMP ↔ yfinance 2자 비교로 제한되며, 두 소스 모두 Yahoo Finance 파생
가능성이 있어 진정한 독립 검증이 아님.

**조치:** Tier 1 종목의 핵심 수치는 투자자가 10-K 원본에서 수동 재확인할 것.
설계 문서 §3.2 "Phase 1 한계 고지" 참조.
