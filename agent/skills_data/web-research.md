---
name: web-research
description: Finding a fact on the web - search to find the page, then read the page.
triggers: [search, fetch, web, wikipedia, online, internet, source, cite]
phases: [plan, execute, fix]
---

## Search finds the page; the page holds the answer

A search result's snippet is a few hundred characters cut from a page, and the
fact you need is usually somewhere else on that page. One or two searches to
find the page, then `fetch_url` the most promising result and read it. On this
harness's research questions, runs that answered wrongly made 13.5 searches and
opened their first page after 4.8; runs that answered the same questions right
made 8 and opened a page after 2.7.

## Before the first search

Write down the facts the question needs, in order, and what kind of page owns
each one:

- a person, place, organisation or work: its Wikipedia article
  (`https://en.wikipedia.org/wiki/<Title>`) or its official site;
- a paper: its journal page, its arXiv abstract, or `https://doi.org/<doi>`;
- a page as it was on a date, or a number "as of" a date: the Wayback Machine,
  `https://web.archive.org/web/<YYYYMMDD>/<url>`, which serves the copy nearest
  that date;
- a statistic: the agency or organisation that publishes it.

## Queries

- Short: the entity's name and the one or two words that pick out the fact,
  three to eight words. Not the question itself.
- Quote exact titles and phrases: `"Diamond photonic crystal slab"`.
- `site:` sends a query to the page's owner: `site:nature.com 2012 plasmon`.
- When a search finds nothing useful, change the entity or the source, not the
  wording: a rephrased query returns the same pages.

## Reading

- Ask `fetch_url` for more (`max_chars` up to 32000) when the fact sits deep in
  a long page.
- Take the fact from the page's text, not from a snippet or from memory, and
  keep the URL it came from in your notes.

## The answer

Write the answer in exactly the form the task asks for. Where it came from
belongs in your notes or summary, never in the answer itself.
