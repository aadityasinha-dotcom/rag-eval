.PHONY: dev down fetch index eval lint test chunks grep golden-check

UV ?= uv

dev:              ## postgres 16 + pgvector via docker compose
	docker compose up -d --wait

down:
	docker compose down

fetch:            ## download corpus per scripts/sources.txt
	python3 scripts/fetch_corpus.py

chunks:           ## browse chunks: make chunks DOC=doc3 [STRATEGY=fixed512]
	$(UV) run python -m ingest.chunk --strategy $(or $(STRATEGY),fixed512) --show $(DOC)

grep:             ## find chunk ids containing a phrase: make grep Q="NOWAIT"
	$(UV) run python -m ingest.chunk --strategy $(or $(STRATEGY),fixed512) --grep "$(Q)"

golden-check:     ## validate evals/golden.jsonl against current chunk ids
	$(UV) run python -m evals.golden --check --strategy $(or $(STRATEGY),fixed512)

index:            ## (step 4) ingest corpus with a chunking strategy
	@echo "ingest/index.py not written yet (build order step 4)"; exit 1

eval:             ## (step 7) run golden set against a config
	@echo "evals/run.py not written yet (build order step 7)"; exit 1

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

test:
	$(UV) run pytest -q
