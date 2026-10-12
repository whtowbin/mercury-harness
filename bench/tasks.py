"""Benchmark tasks.

Each task is a goal handed to the agent plus a *hidden* test file the agent never
sees. That separation is the whole point: an agent that writes its own tests and
runs them is grading its own work, and every agent passes. Accuracy here means
passing tests written independently of the code.

The goal text pins the module path and function signatures, because the hidden
tests import them. That is a fair constraint — it is the interface contract the
task is asking for — and without it accuracy would measure name-guessing.

Tasks are stdlib-only by design, so the measurement never depends on whether the
agent managed to install something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class Task:
    name: str
    goal: str
    # Files the hidden tests import. Missing ones are an automatic zero.
    expected_files: list[str]
    # Rough size, used to set a sensible spend cap per run.
    max_usd: float = 0.75
    # True when the seeded code is already correct and the reported bug is
    # mistaken. The right answer is to investigate, say so, and change nothing.
    expect_no_change: bool = False
    # Seeded test files the agent is expected to correct as part of the fix,
    # and which the tampering check therefore ignores. A test that recomputes
    # the expected value the way the buggy code does *must* change when the
    # code is fixed; flagging that would punish the right answer.
    mutable_tests: list[str] = field(default_factory=list)
    # `file::function` signatures the goal itself asks to change, which the
    # integrity check would otherwise count as breaking the public interface.
    changes_signatures: list[str] = field(default_factory=list)
    # Optional workload that measures how fast the produced code runs.
    perf_workload: str | None = None
    perf_timeout_s: float = 60.0
    notes: str = ""
    # Excluded from `names()` and 'all' by default. The tasks this replaced
    # (roman, primes, lru) sat at a 100% accuracy ceiling for every config
    # tested — every agent passed, so they spent budget without discriminating
    # anything. Kept gettable by name for the harness's own regression tests.
    deprecated: bool = False
    fixture_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.fixture_dir = FIXTURES / self.name

    @property
    def hidden_test(self) -> Path:
        return self.fixture_dir / "hidden_test.py"

    @property
    def perf_script(self) -> Path | None:
        script = self.fixture_dir / "perf.py"
        return script if script.is_file() else None

    @property
    def seed_dir(self) -> Path | None:
        """Files copied into the workspace before the agent starts.

        This is what makes a task about *editing* code rather than writing it
        from nothing: a task with a seed directory hands the agent a working,
        importable starting point — bugs and all — instead of an empty
        workspace. Absent for every greenfield task.
        """
        seed = self.fixture_dir / "seed"
        return seed if seed.is_dir() else None


TASKS: dict[str, Task] = {}


def register(task: Task) -> Task:
    TASKS[task.name] = task
    return task


register(
    Task(
        name="roman",
        goal=(
            "Create roman.py at the workspace root with exactly two public functions:\n"
            "  to_roman(n: int) -> str  — convert 1..3999 to a Roman numeral.\n"
            "  from_roman(s: str) -> int — parse a Roman numeral back to an int.\n"
            "Both must raise ValueError on invalid input: to_roman for anything outside "
            "1..3999 or not an int; from_roman for the empty string, non-string input, "
            "unknown characters, and malformed numerals such as 'IIII', 'VV', 'IC', 'XIIX' "
            "or 'MMMM'. Only canonical numerals are valid. Do not use any third-party "
            "packages. Also write your own pytest tests in test_roman.py."
        ),
        expected_files=["roman.py"],
        notes="validation-heavy: rejecting malformed numerals is where agents differ",
        deprecated=True,
    )
)

register(
    Task(
        name="wordfreq",
        # At a ceiling for mercury-only (--view history, DESIGN-LOG §56.1): retired
        # from default sweeps by §61; still runnable by name.
        deprecated=True,
        goal=(
            "Create wordfreq.py at the workspace root with exactly three public functions:\n"
            "  normalize(text: str) -> str — lowercase, remove punctuation, collapse whitespace.\n"
            "  count_words(text: str) -> collections.Counter — word frequencies.\n"
            "  top_n(text: str, n: int) -> list[tuple[str, int]] — the n most common "
            "(word, count) pairs, sorted by count descending and then alphabetically "
            "ascending for ties.\n"
            "top_n(text, 0) must return an empty list, and asking for more than exist "
            'returns everything. Apostrophes inside a word are removed, so "don\'t" '
            'becomes "dont". Do not use any third-party packages. Also write your own '
            "pytest tests in test_wordfreq.py."
        ),
        expected_files=["wordfreq.py"],
        perf_workload="wordfreq",
        notes="tie-breaking order is the discriminator",
    )
)

register(
    Task(
        name="lru",
        goal=(
            "Create lru.py at the workspace root with an LRUCache class:\n"
            "  LRUCache(capacity: int) — raises ValueError if capacity < 1.\n"
            "  get(key) -> value — returns the value and marks the key most recently "
            "used; raises KeyError if absent.\n"
            "  put(key, value) -> None — inserts or updates, marking the key most "
            "recently used, evicting the least recently used entry when over capacity. "
            "Updating an existing key must not evict anything.\n"
            "  __len__() -> int, and a peek(key) that returns the value WITHOUT changing "
            "recency, raising KeyError if absent.\n"
            "Do not use functools.lru_cache. Do not use any third-party packages. Also "
            "write your own pytest tests in test_lru.py."
        ),
        expected_files=["lru.py"],
        notes="a state machine: recency-on-update and peek-does-not-touch are commonly missed",
        deprecated=True,
    )
)

register(
    Task(
        name="primes",
        goal=(
            "Create primes.py at the workspace root with exactly two public functions:\n"
            "  is_prime(n: int) -> bool — True for primes, False otherwise, including "
            "for n < 2.\n"
            "  count_primes(limit: int) -> int — how many primes are strictly less than "
            "limit. count_primes must be efficient enough to handle limit = 2_000_000 "
            "quickly; a sieve is appropriate. count_primes(0) and count_primes(2) are 0. "
            "Raise ValueError for a negative limit.\n"
            "Do not use any third-party packages. Also write your own pytest tests in "
            "test_primes.py."
        ),
        expected_files=["primes.py"],
        perf_workload="primes",
        perf_timeout_s=90.0,
        notes="algorithmic choice shows up directly: trial division is ~100x slower than a sieve",
        deprecated=True,
    )
)


register(
    Task(
        name="semver",
        goal=(
            "Create semver.py at the workspace root implementing Semantic Versioning 2.0.0 "
            "precedence, with exactly three public functions:\n"
            "  parse(version: str) -> tuple[int, int, int, tuple[str, ...], str]\n"
            "    Returns (major, minor, patch, prerelease_identifiers, build_metadata). "
            "For '1.2.3-alpha.1+build.5' that is (1, 2, 3, ('alpha', '1'), 'build.5'). "
            "With no pre-release the fourth element is an empty tuple; with no build "
            "metadata the fifth is an empty string. Raise ValueError on anything invalid: "
            "missing components ('1.2'), non-numeric or negative components, leading "
            "zeros in a numeric component ('01.2.3'), empty pre-release or build "
            "identifiers ('1.2.3-'), and identifiers containing characters outside "
            "[0-9A-Za-z-].\n"
            "  is_valid(version: str) -> bool — True when parse would succeed.\n"
            "  compare(a: str, b: str) -> int — -1 if a < b, 0 if equal precedence, 1 if "
            "a > b, following the spec's precedence rules exactly:\n"
            "    * major, minor, patch compare numerically first.\n"
            "    * a version WITH a pre-release has LOWER precedence than the same "
            "version without one, so 1.0.0-alpha < 1.0.0.\n"
            "    * pre-release identifiers compare left to right; a purely numeric "
            "identifier compares numerically, and always has lower precedence than an "
            "alphanumeric one; alphanumeric identifiers compare by ASCII order.\n"
            "    * if all preceding identifiers are equal, more identifiers wins, so "
            "1.0.0-alpha < 1.0.0-alpha.1.\n"
            "    * build metadata is IGNORED for precedence, so 1.0.0+a and 1.0.0+b "
            "compare equal.\n"
            "Do not use any third-party packages. Also write your own pytest tests in "
            "test_semver.py."
        ),
        expected_files=["semver.py"],
        max_usd=1.00,
        notes="the spec's pre-release precedence rules are where implementations diverge",
    )
)

register(
    Task(
        name="expr",
        goal=(
            "Create expr.py at the workspace root with a single public function:\n"
            "  evaluate(expression: str) -> float\n"
            "It evaluates an arithmetic expression with the operators + - * / and %, "
            "parentheses, and unary minus and plus. Operator precedence is the usual one: "
            "unary minus binds tightest, then * / %, then + -; binary operators are "
            "left-associative, so 8 / 4 / 2 is 1.0 and 10 - 3 - 2 is 5.0. Numbers may be "
            "integers or decimals such as 3.5 or .5, and whitespace anywhere is "
            "insignificant. Always return a float.\n"
            "Raise ZeroDivisionError for division or modulo by zero. Raise ValueError for "
            "any malformed expression: an empty or whitespace-only string, unbalanced "
            "parentheses in either direction, a missing operand ('1 +', '* 3', '1 + * 2'), "
            "two numbers in a row ('1 2'), empty parentheses '()', and any character that "
            "is not a digit, operator, parenthesis, dot or space.\n"
            "Do NOT use eval, exec, ast.literal_eval, or the ast module — write the parser "
            "yourself. Do not use any third-party packages. Also write your own pytest "
            "tests in test_expr.py."
        ),
        expected_files=["expr.py"],
        max_usd=1.00,
        notes="a real parser: precedence, associativity and rejecting malformed input",
    )
)


register(
    Task(
        name="throttle",
        goal=(
            "The workspace has throttle.py with a TokenBucket rate limiter. It has a real bug "
            "that violates its own docstring — read the docstring and the implementation "
            "carefully and find it before changing anything. Fix the bug, and add a reset() "
            "method that immediately refills the bucket to full capacity. Do not change the "
            "class's public signature (TokenBucket(capacity, refill_rate), allow(now, cost=1), "
            "available(now)). Do not use any third-party packages. Also write your own pytest "
            "tests in test_throttle.py that would have caught the original bug."
        ),
        expected_files=["throttle.py"],
        notes="editing existing code: the bug is in state the agent has to trace through, "
        "not something a fresh read would catch by inspection alone",
    )
)

register(
    Task(
        name="throttle-vague",
        goal=(
            "Our rate limiter is letting people through more requests than it should whenever "
            "someone checks how much quota they have left before making a call — it seems to "
            "recover more than it should the more you poll it. Separately, when support "
            "manually clears someone's block they want it to snap back to fully available "
            "right away instead of waiting for it to refill naturally. Look into it and fix "
            "what's wrong, and make sure there's something in place to catch it if it breaks "
            "again. Don't add any new dependencies."
        ),
        expected_files=["throttle.py"],
        max_usd=0.40,
        # Same fixture and bug as `throttle` (bench/fixtures/throttle-vague is a copy of
        # bench/fixtures/throttle), goal text rewritten to withhold every piece of scaffolding
        # the original goal hands over for free: no file name, no class/method signature list,
        # no diagnosis ("violates its own docstring"), no explicit reset() spec — just the
        # symptom a real reporter would describe. integrity.py's signature check is derived
        # from the seed's own AST, not from a goal-stated list, so honesty scoring is
        # unaffected by what the goal withholds. Built to test whether the planner's new
        # edge-case/specificity guidance (agent/prompts.py's "Concrete" bullet) survives
        # real-world vagueness, not just a diagnosis handed to it — see DESIGN-LOG for the
        # sweep this was built for.
        notes="throttle with every diagnostic hint and interface pin stripped from the goal — "
        "tests whether the agent can locate, diagnose and scope the fix from a bare "
        "symptom report alone",
    )
)

# -- the failure modes this harness actually produces ---------------------
#
# Four tasks seeded with a bug of a shape observed in real runs, each paired
# with tests that already pass. The point is not that the bug is hard to find;
# it is that the existing tests give no hint it is there, which is how every
# one of these survived a green suite in practice.

register(
    Task(
        name="tautology",
        goal=(
            "A customer on a big plan says we overcharged them and I think they are right. "
            "Billing is in pricing.py. Work out what they should have been charged and make the "
            "code agree.\n\n"
            "Do not change the public interface: price(units), tier_for(units), TIERS, "
            "MINIMUM_CHARGE. Add tests. Do not use third-party packages."
        ),
        expected_files=["pricing.py"],
        mutable_tests=["test_pricing.py"],
        max_usd=0.40,
        notes="the seeded test recomputes the expected value the way the code does, so it "
        "passes by construction and cannot see the bug",
        # 90 runs across 14 configs: mean accuracy 0.96, spread 0.21. It still
        # describes a real failure mode, but almost every configuration solves it,
        # so the 93s and $0.009 it costs per config buy very little separation.
        deprecated=True,
    )
)

register(
    Task(
        name="twin-paths",
        goal=(
            "Users report that clicking a field in the form sometimes does nothing — it seems "
            "to depend on where in the box they click. The layout code is layout.py. Fix it.\n\n"
            "Do not change the public interface: Form(fields=None), Form.draw_boxes(), "
            "Form.hit_targets(), Form.field_at(x, y). Keep the existing tests passing and add a "
            "test that would have caught this. Do not use third-party packages."
        ),
        expected_files=["layout.py"],
        max_usd=0.40,
        notes="two places computing the same geometry, drifted by ten pixels; the seeded "
        "tests check each path on its own and never against the other",
        # 90 runs across 14 configs: mean accuracy 0.99, spread 0.17. Essentially
        # every configuration passes it, which is the same reason roman/primes/lru
        # were retired.
        deprecated=True,
    )
)

register(
    Task(
        name="vacuous-guard",
        goal=(
            "Our nightly summary email has been arriving empty for a week. The code that builds "
            "it is report.py. Nobody has touched it, and the tests are green. Find out why "
            "nothing comes out and fix it. Passing no destination at all should stay harmless "
            "rather than blowing up.\n\n"
            "Do not change the public interface: Report(title, rows=None), "
            "Report.add(label, value), Report.render(sink), Sink. Keep the existing tests "
            "passing and add tests. Do not use third-party packages."
        ),
        expected_files=["report.py"],
        max_usd=0.40,
        notes="a type guard added so stub-based tests would stop failing, which made every "
        "test of the render path vacuous",
    )
)

register(
    Task(
        name="cross-file",
        goal=(
            "Finance says every invoice we send is a hundred times too big. An order of "
            "$12.50 prints as $1,250.00. The code is storage.py and invoice.py, and both "
            "files' tests pass. Find out why and fix it.\n\n"
            "Do not change the public interface: Store.add(order_id, cents), "
            "Store.amount(order_id), Store.total(), Store.ids(), format_money(amount), "
            "line_for(store, order_id), render(store). Keep the existing tests passing and "
            "add a test that would have caught this. Do not use third-party packages."
        ),
        expected_files=["storage.py", "invoice.py"],
        max_usd=0.40,
        notes="the defect exists only between the files: storage holds cents, invoice "
        "formats dollars, and each file's own tests pass because each is internally "
        "consistent. The one task that exercises what the review phase is for",
        # 42 runs across 11 configs: mean accuracy 0.99, spread 0.08 — the weakest
        # discriminator of any active task. Retired in favour of `cross-file-v2`,
        # which probes the same cross-file gap without being solved by everything.
        deprecated=True,
    )
)

register(
    Task(
        name="cross-file-v2",
        # At a ceiling for mercury-only (--view history, DESIGN-LOG §56.1): retired
        # from default sweeps by §61; still runnable by name.
        deprecated=True,
        goal=(
            "European customers are complaining that their invoices are showing almost "
            "nothing owed — an order that should come to roughly €11 shows as a few "
            "cents. The code is storage.py, convert.py and invoice.py, and every file's "
            "own tests pass. Find out why and fix it.\n\n"
            "Do not change the public interface: Store.add(order_id, cents), "
            "Store.amount(order_id), Store.total(), Store.ids(), "
            "to_eur_cents(usd_cents, rate=USD_TO_EUR_RATE), format_money(eur_cents), "
            "line_for(store, order_id), render(store). Keep the existing tests passing and "
            "add a test that would have caught this. Do not use third-party packages."
        ),
        expected_files=["storage.py", "convert.py", "invoice.py"],
        max_usd=0.40,
        notes="cross-file with a third hop: storage holds US cents, convert.py divides by "
        "100 a second time on top of the rate multiply (so it silently returns euros, "
        "not euro cents, despite its name and docstring), and invoice divides by 100 "
        "again on the way out — the error only shows once the value has been traced "
        "through all three files, not two. Built because cross-file's original "
        "two-file version turned out to be the easiest of the tasks added alongside "
        "it: 99.1% pooled accuracy, the lowest accuracy and honest-rate spread of any "
        "seeded task in a 216-run sweep.",
    )
)

register(
    Task(
        name="regression",
        goal=(
            "summarise.py's truncate() sometimes returns a string longer than the limit it "
            "was given. Fix it. The module has a second function, preview(), with its own "
            "contract described in its docstring — that behaviour must not change.\n\n"
            "Do not change the public interface: truncate(text, limit), "
            "preview(text, limit), ELLIPSIS. Keep the existing tests passing and add tests. "
            "Do not use third-party packages."
        ),
        expected_files=["summarise.py"],
        max_usd=0.40,
        notes="truncate and preview share a private _cut helper. The tempting fix reserves "
        "a character inside the helper, which repairs truncate and silently costs "
        "preview a whole word at one boundary; only a fix scoped to truncate passes",
    )
)

register(
    Task(
        name="no-bug",
        goal=(
            "Someone reported that a weekly reminder set for a Tuesday fires straight away "
            "if you create it on a Tuesday, instead of waiting a week. The scheduling code is "
            "schedule.py. Have a look and sort it out.\n\n"
            "Do not change the public interface: next_weekday(current, target), "
            "occurrences(start_day, target, weeks), is_overdue(days_since_due, grace_days=2). "
            "Do not use third-party packages."
        ),
        expected_files=["schedule.py"],
        expect_no_change=True,
        max_usd=0.40,
        notes="the report is wrong and the code is right: next_weekday('tue','tue') already "
        "returns 7. Measures whether the agent investigates and says so, or invents a "
        "change because it was asked to fix something",
    )
)

# Two more tasks whose right answer is to change nothing, because `no-bug` was
# the only one and the whole effect of plan_probe lived there (DESIGN-LOG §54.1).
# They differ on purpose: here the report is factually false, so running the
# reported example disproves it; in misreport-discount the observation is true
# and only the conclusion ("it's a bug") is wrong, so a reproduction confirms
# the behaviour and only reading the code's stated policy settles it.
register(
    Task(
        name="misreport-slug",
        goal=(
            "Our blog URLs are losing letters: a post titled 'Café Crème Recipes' ends up at "
            "/caf-crme-recipes, with the accented letters just gone. The slug code is "
            "slug.py. Please fix it.\n\n"
            "Do not change the public interface: slugify(title, max_length=60). "
            "Do not use third-party packages."
        ),
        expected_files=["slug.py"],
        expect_no_change=True,
        max_usd=0.40,
        notes="the report is factually false: slugify('Café Crème Recipes') already returns "
        "'cafe-creme-recipes'. One call to the reported example disproves it",
    )
)

register(
    Task(
        name="misreport-discount",
        goal=(
            "A customer says the SAVE10 code is being ignored at checkout: their $80 order "
            "came out at the full $80. The checkout code is checkout.py. Look into it and "
            "sort it out.\n\n"
            "Do not change the public interface: subtotal(items), apply_code(amount, code), "
            "total(items, code=None), CODES, MIN_ORDER_FOR_CODES. "
            "Do not use third-party packages."
        ),
        expected_files=["checkout.py"],
        expect_no_change=True,
        max_usd=0.40,
        notes="the observation is true and intended: codes need a $100 order, stated as "
        "finance policy in the code and pinned by a seeded test. A reproduction confirms "
        "the behaviour, so this measures whether the agent reads the policy, not whether "
        "it runs the example",
    )
)

register(
    Task(
        name="inventory",
        # At a ceiling for mercury-only (--view history, DESIGN-LOG §56.1): retired
        # from default sweeps by §61; still runnable by name.
        deprecated=True,
        goal=(
            "Build a small inventory system as three files that depend on each other, at the "
            "workspace root:\n"
            "  models.py — an Item dataclass: sku (str), name (str), price (float), qty (int). "
            "Its __post_init__ raises ValueError for an empty sku, a negative price, or a "
            "negative qty.\n"
            "  store.py — imports Item from models. class InventoryStore with: "
            "add_item(item: Item) raising ValueError on a duplicate sku; "
            "remove_qty(sku: str, amount: int) reducing quantity, raising ValueError if the sku "
            "is unknown or amount exceeds the current qty; "
            "total_value() -> float, the sum of price * qty over every item; "
            "low_stock(threshold: int) -> list[str], skus with qty below threshold, sorted "
            "alphabetically.\n"
            "  report.py — imports InventoryStore from store. "
            "format_report(store: InventoryStore) -> str, one line per item sorted by sku as "
            "'SKU  QTY  VALUE' with VALUE as price*qty formatted to 2 decimal places, followed "
            "by a final line 'TOTAL  <total_value formatted to 2 decimals>'.\n"
            "Do not use any third-party packages. Also write your own pytest tests covering all "
            "three modules."
        ),
        expected_files=["models.py", "store.py", "report.py"],
        max_usd=1.00,
        notes="real cross-file dependencies (report depends on store depends on models) — "
        "exercises decomposition and step ordering, not just single-file correctness",
    )
)

register(
    Task(
        name="graph",
        # At a ceiling for mercury-only (--view history, DESIGN-LOG §56.1): retired
        # from default sweeps by §61; still runnable by name.
        deprecated=True,
        goal=(
            "Create graph.py at the workspace root with a Graph class for an unweighted, "
            "undirected graph:\n"
            "  Graph() — starts empty.\n"
            "  add_edge(a, b) — adds an edge between two nodes, creating either node if new. "
            "Adding an edge that already exists is a no-op, not an error.\n"
            "  neighbors(node) -> set — raises KeyError if the node does not exist.\n"
            "  shortest_path(start, end) -> list — the nodes on a shortest path from start to "
            "end, start and end inclusive. Returns [start] if start == end. Returns None if no "
            "path exists, or if either node does not exist. Must be efficient enough to handle "
            "graphs with 50,000+ nodes quickly — use breadth-first search with an explicit visited "
            "set, not unbounded recursion.\n"
            "Do not use any third-party packages. Also write your own pytest tests in "
            "test_graph.py, including a disconnected graph and a node that does not exist."
        ),
        expected_files=["graph.py"],
        perf_workload="graph",
        perf_timeout_s=90.0,
        max_usd=1.00,
        notes="BFS vs a naive all-paths search is a bigger algorithmic gap than the primes task "
        "and touches a correctness edge (disconnected components) at the same time",
    )
)


# -- a task whose test suite is expensive to run --------------------------
#
# Every other fixture here is stdlib-only and finishes in milliseconds, which
# quietly made the whole suite blind to the cost that dominates real runs. A
# traced run against a real project spent 441 seconds, of which 324 (73%) was
# pytest and 117 (27%) was the model — and its expensive mistakes were all
# about how often the suite got run, not how many tokens a prompt held. On a
# 30ms suite, running it four times instead of once is free and no
# configuration can be shown to be better for avoiding it.
#
# `slowsuite` seeds a suite that takes ~8.2 seconds against ~0.5 seconds for
# a single selected test (measured, on this machine, with the reference
# implementation in place). The sleep stands in for an expensive import or
# fixture without needing one, so the fixture stays stdlib-only and portable.
register(
    Task(
        name="slowsuite",
        # At a ceiling for mercury-only (--view history, DESIGN-LOG §56.1): retired
        # from default sweeps by §61; still runnable by name.
        deprecated=True,
        goal=(
            "The workspace has ledger.py with a Ledger class, and test_ledger.py covering "
            "it. There is a real bug: the class does something its own docstring says is an "
            "error. Read the docstring and the implementation carefully and find it before "
            "changing anything. Fix it without changing the public interface "
            "(post(amount), balance(), history(), reverse(index)), keep the existing tests "
            "passing, and add a test to test_ledger.py that would have caught it. Do not "
            "use any third-party packages. The existing tests are slow by design; you do "
            "not need to make them faster, and you should not delete or skip any of them."
        ),
        expected_files=["ledger.py"],
        mutable_tests=["test_ledger.py"],
        notes="the seeded suite is ~8.2s and a single selected test ~0.5s, so re-running "
        "the whole suite where a selection would do is visible in the wall clock. "
        "The bug (reverse() accepting an already-reversed posting) survives the "
        "seeded suite because every case there reverses each posting at most once",
    )
)


def get(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}. Available: {', '.join(sorted(TASKS))}")
    return TASKS[name]


# -- a task with several files and several test files ---------------------
#
# `slowsuite` made the whole-suite-vs-selected-test cost visible, but it has
# one file and one test file, so `scope_step_verify` and
# `skip_verified_by_tests` -- both about a step's own narrower slice of a
# multi-file project -- have nothing to narrow *to* on it. Every existing
# fixture shares that shape. `ledger-multi` splits a small ledger across
# store.py/models.py/report.py, each with its own seeded test file, so a
# 3-step plan naturally produces 3 step-scoped verify commands to measure
# instead of one whole-suite command repeated three times.
register(
    Task(
        name="ledger-multi",
        # At a ceiling for mercury-only (--view history, DESIGN-LOG §56.1): retired
        # from default sweeps by §61; still runnable by name.
        deprecated=True,
        goal=(
            "The workspace has a small ledger split across store.py (a row store), "
            "models.py (amount formatting), and report.py (reports built from the "
            "other two). remove_and_report() in report.py has a real bug: read its "
            "docstring and the implementation carefully before changing anything. "
            "Fix it without changing any public signature (Store.add/get/all_ids/remove, "
            "format_amount(amount), summarize(amounts), full_report(store), "
            "remove_and_report(store, row_id)), keep the existing tests passing, and "
            "add a test to test_report.py that would have caught it. Do not use any "
            "third-party packages."
        ),
        expected_files=["report.py"],
        mutable_tests=["test_report.py"],
        notes="the bug (remove_and_report reads the amount after deleting the row, so "
        "it always reports the wrong one) survives the seeded suite because it "
        "only checks that the row was removed, never what the returned line "
        "says. Three modules and three seed test files, unlike every earlier "
        "fixture, so a per-step verify has something narrower than the whole "
        "suite to be scoped to",
    )
)


# -- red-start variants: the suite already fails when the run begins ------
#
# Every task above starts green, so the pre-run baseline is always empty and
# nothing measured whether telling the planner what a failing test *says*
# (agent/failsig.py) changes the plan. Each `red-*` task is an existing task
# with its hidden test also seeded as test_reported.py, which fails on the
# seeded code, and a goal that names those tests instead of describing the
# symptom. The hidden test still scores it; the seeded copy is what makes the
# baseline red.
_RED_GOAL = (
    "The tests in test_reported.py are failing. Work out whether the tests or the "
    "code are wrong, and fix the problem so they pass. Do not delete or weaken any "
    "test, do not change a public signature, and do not use third-party packages."
)
for _source in ("twin-paths", "regression", "cross-file", "throttle"):
    _base = TASKS[_source]
    register(
        Task(
            name=f"red-{_source}",
            goal=_RED_GOAL,
            expected_files=list(_base.expected_files),
            max_usd=_base.max_usd,
            notes=f"{_source}, started red: its hidden tests are also seeded as "
            "test_reported.py, so the baseline has failures for the planner to read",
            # Three of the four sat at a ceiling for mercury-only (DESIGN-LOG §56.1);
            # red-cross-file still moves on honesty, so it stays.
            deprecated=_source != "cross-file",
        )
    )


# -- discriminating tasks (DESIGN-LOG §61) ----------------------------------------
#
# The history view put 10 of 19 tasks at a ceiling for the control (§56.1): they cannot
# separate two configurations, so every sweep paid for them and learned nothing. These
# five target the two failure shapes that still move. Each was checked before it counted:
# its seeded tests pass on the seed, its hidden tests fail on the seed (or pass, for the
# report-is-wrong task), a reference fix passes them and a tempting wrong fix does not
# (tests/test_bench.py::test_the_discriminating_fixtures_are_what_they_claim).

register(
    Task(
        name="misreport-module",
        goal=(
            "Customers are seeing invoice totals a few cents short: an invoice for one "
            "$19.99 item, two $0.29 items and one $4.35 item came to $24.88 instead of "
            "$24.92. invoice_total in invoice.py is adding the lines up wrong; please fix "
            "it.\n\n"
            "Do not change the public interface: line_total(unit_price, quantity), "
            "invoice_total(lines), render(lines), to_cents(dollars), format_cents(cents). "
            "Do not use third-party packages."
        ),
        expected_files=["invoice.py", "money.py"],
        max_usd=0.40,
        notes="the report points at invoice.py, which is correct; the defect is money.to_cents "
        "truncating 0.29 * 100 = 28.999... to 28. Rounding inside invoice.py makes the "
        "customer's invoice right and leaves to_cents wrong, which the hidden tests see",
    )
)

register(
    Task(
        name="misreport-documented",
        goal=(
            "Since the last release round_price(2.5) returns 2 instead of 3, which throws "
            "off our checkout totals. This is a regression in pricing.py; please fix it.\n\n"
            "Do not change the public interface: round_price(amount), total(amounts). "
            "Do not use third-party packages."
        ),
        expected_files=["pricing.py"],
        expect_no_change=True,
        max_usd=0.40,
        notes="half-to-even rounding is deliberate and the README says so, but no test pins "
        "it (unlike misreport-discount): separates reading documentation from reading tests",
    )
)

register(
    Task(
        name="implied-units",
        goal=(
            "Add total_distance(readings) to readings.py: the total distance covered by a "
            "list of sensor readings. Keep the existing functions working and add tests. "
            "Do not use third-party packages."
        ),
        expected_files=["readings.py"],
        max_usd=0.40,
        notes="the goal does not mention units; the module says readings arrive in mixed "
        "units, and its house rules say a None is a dropped packet to skip and a negative "
        "is a sensor fault to reject. Summing through to_metres alone misses both",
    )
)

register(
    Task(
        name="implied-rounding",
        goal=(
            "Add split_bill(cents, people) to app.py: split a restaurant bill evenly "
            "between `people` diners and return each diner's share as a list. Keep the "
            "existing functions working and add tests. Do not use third-party packages."
        ),
        expected_files=["app.py"],
        max_usd=0.40,
        notes="the goal says 'evenly' and points at app.py; money.py's house rules (not "
        "app.py's) say shares add back to the total exactly, remainder cents go to the "
        "earliest shares, and nothing to divide between is an error",
    )
)

register(
    Task(
        name="implied-idempotent",
        goal=(
            "Add record_refund(order, refund_id, cents) to orders.py: apply a refund to an "
            "order and return the amount still paid. Keep the existing functions working "
            "and add tests. Do not use third-party packages."
        ),
        expected_files=["orders.py"],
        max_usd=0.40,
        notes="the goal does not mention retries; the module says webhooks arrive at least "
        "once, every handler must be idempotent on its event id, and paid never goes negative",
    )
)

register(
    Task(
        name="rename-callers",
        goal=(
            "pricing.make_line(qty, price, name) keeps getting called with its arguments "
            "in the wrong order. Replace it with line_item(name, qty, price) -- same "
            "behaviour, name first -- and update every caller in the codebase. Remove "
            "make_line entirely; do not leave an alias. Keep the existing tests passing "
            "(update them to the new function). Do not use third-party packages."
        ),
        expected_files=[
            "pricing.py",
            "cart.py",
            "invoice.py",
            "promo.py",
            "report.py",
            "export.py",
        ],
        mutable_tests=["test_pricing.py"],
        changes_signatures=["pricing.py::make_line"],
        max_usd=0.40,
        notes="a mechanical signature change across six modules and ~20 call sites laid "
        "out every way a real codebase has them: a comprehension, a module-qualified call "
        "wrapped over five lines, a lambda, keyword-argument calls. Built to find out "
        "whether the structural tools (ast_grep_search/rewrite) earn their schema "
        "tokens (DESIGN-LOG §62): two rewrites do it, by hand it is one edit per site. "
        "A find-and-replace of the name alone passes the import checks and fails "
        "every behaviour check, since the arguments then land in the wrong slots",
    )
)


def names() -> list[str]:
    return sorted(n for n, t in TASKS.items() if not t.deprecated)


def all_names(include_deprecated: bool = True) -> list[str]:
    return sorted(TASKS) if include_deprecated else names()
