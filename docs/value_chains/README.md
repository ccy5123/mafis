# Manual Value Chain Documents

Phase 1 stores hand-written value chain documents here, one Markdown file per ticker
(e.g. `NVDA.md`). Each document is injected into the Analyst's prompt as a block
(no RAG retrieval), per design-v2.2 re-review High #5.

## Template

```markdown
# {Ticker} Value Chain

## 상류 (공급업체)
- Company A — what they supply, criticality, switching cost
- ...

## 동급 (경쟁사)
- Company B — area of competition, relative strength
- ...

## 하류 (고객사)
- Company C — share of revenue (if known), concentration risk
- ...

## 인프라 / 보조
- Power / cooling / logistics / regulatory

## 취약 고리 (Skeptic 힌트)
- Single-source dependencies
- Geopolitical chokepoints
- ...
```

## Guidance

- 5~10 entries per section is enough for Phase 1
- Sources must be cited (10-K page, earnings call date, news URL)
- Revisit after each quarterly earnings call
