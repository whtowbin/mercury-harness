"""--answer-command: a real subprocess answers the run's questions."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from agent.interact import Interactor, command_asker


def _script(tmp_path: Path, body: str) -> str:
    path = tmp_path / "answer.py"
    path.write_text(body)
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(path))}"


def test_the_command_gets_the_question_as_json_and_its_stdout_is_the_answer(tmp_path):
    command = _script(
        tmp_path,
        "import json, sys\n"
        "q = json.load(sys.stdin)\n"
        "print(f\"{q['question']} -> {q['options'][1]} (rec {q['recommended']})\")\n",
    )
    interactor = Interactor(ask=command_asker(command))

    answer = interactor.ask("Which cache?", ["local", "shared"], "local")

    assert answer == "Which cache? -> shared (rec local)"
    assert interactor.answered == 1


def test_a_failing_or_silent_command_falls_back_to_no_answer(tmp_path):
    failing = Interactor(ask=command_asker(_script(tmp_path, "import sys; sys.exit(3)\n")))
    assert failing.ask("Anything?").startswith("(no answer available: the answer command")

    silent = Interactor(ask=command_asker(_script(tmp_path, "pass\n")))
    assert silent.ask("Anything?").startswith("(no answer available")

    missing = Interactor(ask=command_asker(str(tmp_path / "does-not-exist")))
    assert missing.ask("Anything?").startswith("(no answer available")


def test_a_hung_command_costs_one_question_not_the_run(tmp_path):
    hung = command_asker(_script(tmp_path, "import time; time.sleep(30)\n"), timeout_s=0.5)
    assert hung("Anything?").startswith("(no answer available: the answer command failed")


def test_the_question_limit_still_applies(tmp_path):
    log = tmp_path / "asked.jsonl"
    command = _script(
        tmp_path,
        f"import sys; open({str(log)!r}, 'a').write(sys.stdin.read() + '\\n'); print('ok')\n",
    )
    interactor = Interactor(ask=command_asker(command), max_questions=2)

    answers = [interactor.ask(f"q{i}") for i in range(3)]

    assert answers[:2] == ["ok", "ok"]
    assert answers[2].startswith("(no answer available: this run has already asked")
    asked = [json.loads(line)["question"] for line in log.read_text().splitlines()]
    assert asked == ["q0", "q1"]
