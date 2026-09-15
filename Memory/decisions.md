# Decisions

- Zero duplication: identity, MCP client and LLM provider code lives here
  and nowhere else in Thallus.
- Dependency direction is fixed: familiar and servitor depend on
  thallus-core; thallus-core never depends on egregore or scry.
- Servitor consumes identity and MCP only; the provider module has no
  consumer without an LLM role.
- Single main branch; CI green before pushing.
- This repo carries its own Memory v2 pair; cross-component decisions live
  in the Thallus umbrella. Machine-local state stays under ignored Work/.
