.PHONY: dev down fetch index eval lint test chunks grep golden-check search

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

index:            ## ingest corpus per config: make index CONFIG=evals/configs/baseline.yaml
	$(UV) run python -m ingest.index $(CONFIG)

search:           ## dense top-k for a query: make search Q="lock a table" [CONFIG=...]
	$(UV) run python -m retrieval.dense "$(Q)" --config $(or $(CONFIG),evals/configs/baseline.yaml)

eval:             ## run golden set against a config: make eval CONFIG=evals/configs/baseline.yaml [GENERATE=1]
	$(UV) run python -m evals.run $(CONFIG) $(if $(GENERATE),--generate,)

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

test:
	$(UV) run pytest -q
