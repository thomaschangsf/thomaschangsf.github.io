# Chang Family Personal Websites

Four distinct Hugo-powered personal websites sharing a common base theme.

## Structure

```
personal_website/
├── themes/
│   └── shared/          # Base theme (layouts, partials, CSS, JS)
├── sites/
│   ├── thomas_chang/    # Staff ML Engineer      → localhost:1313
│   ├── peter_chang/     # J&J Bio Scientist      → localhost:1314
│   ├── caryse_chang/    # Undergrad Pre-Med      → localhost:1315
│   └── mason_chang/     # High School Student    → localhost:1316
└── Makefile
```

## Quick Start

```bash
# Run a single site in dev mode
make dev-thomas_chang
make dev-peter_chang
make dev-caryse_chang
make dev-mason_chang

# Build all sites for production
make build-all

# Clean all build outputs
make clean
```

## Personas

| Site | Person | Role | Color |
|------|--------|------|-------|
| `thomas_chang` | Thomas Chang | Staff ML Engineer | Dark indigo/cyan |
| `peter_chang` | Peter Chang | J&J Research Scientist | Forest green (light) |
| `caryse_chang` | Caryse Chang | Pre-Med Undergraduate | Rose/blue (light) |
| `mason_chang` | Mason Chang | High School Student | Electric orange/amber |

## Adding Content

Each site's content lives in `sites/<name>/content/`. The homepage is driven by
front matter in `content/_index.md` and params in `hugo.toml` — no template
editing needed for most content changes.

To add a new page (e.g., a publications list):
```bash
/usr/local/bin/hugo new content publications/_index.md --source sites/peter_chang
```

## Customizing a Persona

1. **Content/copy** — edit `sites/<name>/hugo.toml` params
2. **Colors/fonts** — edit `sites/<name>/static/css/persona.css`
3. **Layout overrides** — add files to `sites/<name>/layouts/` (takes precedence over theme)
4. **Shared structure** — edit `themes/shared/layouts/` or `themes/shared/static/`
