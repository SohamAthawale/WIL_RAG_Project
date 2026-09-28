# Thin wrappers over docker compose. `make help` lists the harness commands.
.DEFAULT_GOAL := help
COMPOSE := docker compose

.PHONY: build up models tests score reproduce confusion eval shell down save help

build:      ## build the harness image
	$(COMPOSE) build harness

up:         ## start Ollama with the GPU attached
	$(COMPOSE) up -d ollama

models: up  ## pull nomic-embed-text and qwen2.5:7b-instruct (once)
	$(COMPOSE) run --rm harness models

tests:      ## run all four control suites (offline, no GPU)
	$(COMPOSE) run --rm --no-deps harness tests

score:      ## re-derive retrieval, refusal, grounding and the comparison table
	$(COMPOSE) run --rm --no-deps harness score

reproduce:  ## re-derive them and diff against the published copies
	$(COMPOSE) run --rm --no-deps harness reproduce

confusion:  ## cross-subclass confusion counters (needs Ollama)
	$(COMPOSE) run --rm harness confusion

eval:       ## full generation run over the test collection (slow, needs GPU)
	$(COMPOSE) run --rm harness eval

shell:      ## interactive shell in the harness image
	$(COMPOSE) run --rm --no-deps harness shell

down:       ## stop everything (models are kept in the named volume)
	$(COMPOSE) down

save:       ## export both images to a tarball for transfer
	bash docker/export-image.sh

help:
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-11s\033[0m %s\n", $$1, $$2}'
