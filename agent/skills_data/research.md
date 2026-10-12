---
name: research
description: Finding out how a library or API actually behaves, from the source that owns it rather than from memory.
triggers: [api, docs, documentation, library, changelog, rfc, spec, endpoint, sdk, "how does", "what does", version, upgrade, migrate, deprecated, http, url]
phases: [plan, execute, fix]
---

Most wrong code here is not wrong logic, it is a confidently misremembered
API: a keyword argument that does not exist, a return type that changed two
versions ago, a function that returns `None` in the case you cared about.
Guessing is fast and the guess is often nearly right, which is what makes it
expensive — the failure surfaces three steps later as behaviour nobody
expected.

## Ask the thing itself before you ask the internet

For anything already installed, `inspect_module` is the primary source and it
is one call:

    inspect_module("pathlib", "Path.write_text")
    inspect_module("pygame", "Surface.blit")

It gives the real signature, the docstring and the members, from the version
actually installed in this workspace — which is the version that will run, and
which no web page can tell you. Reach for it before web_search, before
fetch_url, and instead of writing a scratch script to print `dir(x)`.

`pypi_info` answers a different question: does this package exist, what is its
current version, what does it depend on. Use it before adding a dependency,
not to learn an API.

## Primary sources, not write-ups about them

When you do go outside, prefer the source that owns the claim: official docs,
the project's own changelog, the spec, the source file. A blog post explaining
a library is one person's reading of it from some earlier version, and it ages
badly in exactly the places that matter. `web_search` is for *finding* the
primary source; `fetch_url` is for reading it once you know where it is.

Follow a claim back to whoever owns it. "Stack Overflow says X" is not a
finding; "the changelog for 2.4 says X, here is the line" is.

## Bound it, then stop

Research is not the work. Two or three lookups settle almost everything worth
settling, and past that you are reading instead of building. If the answer is
not converging, write down what you found, say plainly what remains unknown,
and make the smallest change that would reveal the answer — a failing test
against the real API is often faster than another page.

## Write down what you learned

A fact discovered in step two and forgotten by step five gets rediscovered at
full price, or worse, guessed at the second time. Use `remember` for anything
a later step would otherwise have to look up again: the real signature, the
version constraint, the behaviour that surprised you, the approach that turned
out not to work. Keep the citation with it — the module and member you
inspected, or the URL — so the next reader can check rather than trust.

If the finding is substantial enough to outlive the run, and the workspace has
somewhere such notes already live, write it there as Markdown instead, and say
in your summary where you put it.
