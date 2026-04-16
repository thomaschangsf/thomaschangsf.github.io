HUGO := /usr/local/bin/hugo
SITES := thomas_chang peter_chang caryse_chang mason_chang

# Ports for each dev server
PORT_thomas_chang  := 1313
PORT_peter_chang   := 1314
PORT_caryse_chang  := 1315
PORT_mason_chang   := 1316

.PHONY: all dev-all build-all clean help $(SITES) build-% dev-%

## help: Show this help
help:
	@echo ""
	@echo "  Chang Family Personal Websites"
	@echo "  ────────────────────────────────────────────"
	@echo "  make dev-thomas_chang    Start thomas_chang dev server (port 1313)"
	@echo "  make dev-peter_chang     Start peter_chang dev server (port 1314)"
	@echo "  make dev-caryse_chang    Start caryse_chang dev server (port 1315)"
	@echo "  make dev-mason_chang     Start mason_chang dev server (port 1316)"
	@echo ""
	@echo "  make build-thomas_chang  Build thomas_chang"
	@echo "  make build-peter_chang   Build peter_chang"
	@echo "  make build-caryse_chang  Build caryse_chang"
	@echo "  make build-mason_chang   Build mason_chang"
	@echo ""
	@echo "  make build-all           Build all sites"
	@echo "  make clean               Remove all public/ directories"
	@echo ""

## dev-<site>: Start Hugo dev server for a single persona (kills existing port first)
dev-%:
	@echo "Checking port $(PORT_$*)..."
	@-lsof -t -i:$(PORT_$*) | xargs kill -9 2>/dev/null || true
	$(HUGO) server --source sites/$* --port $(PORT_$*) --baseURL http://localhost:$(PORT_$*) --appendPort=true --buildDrafts --disableFastRender

## build-<site>: Build a single persona site
build-%:
	$(HUGO) --source sites/$* --minify

## build-all: Build all four sites
build-all: $(addprefix build-,$(SITES))

## clean: Remove all built output
clean:
	@for site in $(SITES); do \
	  rm -rf sites/$$site/public; \
	  echo "Cleaned sites/$$site/public"; \
	done
