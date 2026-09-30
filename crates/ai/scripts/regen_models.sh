#!/usr/bin/env bash
# Regenerate src/models.generated.json from upstream pi's published model catalog.
#
# Upstream no longer commits its catalog data (it is hydrated from live provider APIs at build
# time), so we take it from the published npm package, which ships the built catalog:
#
#   crates/ai/scripts/regen_models.sh            # latest @earendil-works/pi-ai
#   PI_AI_VERSION=0.99.1 crates/ai/scripts/regen_models.sh
#
# Only models whose `api` pie implements (see src/providers/register_builtins.rs) are kept, and
# every kept model must carry the fields `models_generated.rs` requires, so the catalog can
# never make the loader panic. Extra upstream fields (inputLimits, promptCache, ...) are kept
# verbatim; the loader ignores what it does not understand.
set -euo pipefail
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
PKG="@earendil-works/pi-ai"
VERSION="${PI_AI_VERSION:-latest}"
SUPPORTED_APIS="${SUPPORTED_APIS:-anthropic-messages,azure-openai-responses,bedrock-converse-stream,google-generative-ai,google-vertex,mistral-conversations,openai-codex-responses,openai-completions,openai-responses}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

(cd "$WORK" && npm pack --silent "$PKG@$VERSION" >/dev/null && tar xzf ./*.tgz)
RESOLVED="$(node -p "require('$WORK/package/package.json').version")"

SUPPORTED_APIS="$SUPPORTED_APIS" node --input-type=module -e "
const { MODELS } = await import('$WORK/package/dist/models.generated.js');
const supported = new Set(process.env.SUPPORTED_APIS.split(','));
const required = ['id', 'name', 'api', 'provider', 'baseUrl', 'reasoning', 'cost', 'contextWindow', 'maxTokens'];
const costKeys = ['input', 'output', 'cacheRead', 'cacheWrite'];
const out = {};
let kept = 0, skipped = 0;
for (const provider of Object.keys(MODELS).sort()) {
  for (const id of Object.keys(MODELS[provider]).sort()) {
    const m = MODELS[provider][id];
    const ok = supported.has(m.api)
      && required.every((k) => m[k] !== undefined && m[k] !== null)
      && costKeys.every((k) => typeof m.cost[k] === 'number');
    if (!ok) { skipped++; continue; }
    (out[provider] ??= {})[id] = m;
    kept++;
  }
}
process.stdout.write(JSON.stringify(out));
console.error('kept ' + kept + ' models, skipped ' + skipped + ' (unsupported api or missing fields)');
" > "$HERE/src/models.generated.json"

echo "regenerated $HERE/src/models.generated.json from $PKG@$RESOLVED ($(wc -c < "$HERE/src/models.generated.json") bytes)"
