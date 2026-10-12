# Quickstart

This page takes you from nothing to a working install with an API key stored. {doc}`usage`
then covers running the agent.

## Requirements

- macOS or Linux. Windows is not supported: the harness needs POSIX process groups to stop
  runaway commands.
- [git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/).
- Python 3.11 or later (uv fetches one if you have none).
- An API key for [OpenRouter](https://openrouter.ai/) or for Inception's own API.

## Install

The harness is not on PyPI. Install it from GitHub as a tool, which puts `harness`,
`harness-tui` and `bench` on your PATH in their own environment:

```bash
uv tool install git+https://github.com/whtowbin/mercury-harness
harness --version
```

Three extras are optional:

| extra | adds | needed for |
| --- | --- | --- |
| `tui` | the terminal dashboard | `harness-tui`, `harness run --tui` |
| `analysis` | DuckDB | `harness trace sql` |
| `keyring` | the `keyring` library | storing your key in the system keyring on Linux (and Windows, though the harness does not run there) |

Ask for one by name, or several:

```bash
uv tool install 'mercury-harness[tui,analysis] @ git+https://github.com/whtowbin/mercury-harness'
```

Using a feature without its extra prints this command instead of failing obscurely. To update,
add `--force` to the install command. To remove it: `uv tool uninstall mercury-harness`.

## Add an API key

```bash
harness setup
```

`harness setup` asks which provider to use:

- **OpenRouter**: one key covers Mercury and every other model, so it is the choice if you
  might ever pass `--planner some/other-model`.
- **Inception**: Inception's own API, Mercury only. Choosing it also persists
  `mercury_provider=inception`, so Mercury calls go there from then on.

It then reads the key without echoing it, checks it with the provider, and stores it. The check
costs nothing for OpenRouter (it calls the free key endpoint). For Inception it sends one request
capped at one output token, under $0.00001. `--no-check` skips it.

To skip the questions: `harness setup --provider openrouter --store file`.

### Where the key is stored

`--store secure` or `--store file` picks the place. Without the flag the harness uses `secure`
where the machine has a secret store and `file` where it does not, and says so with a hint.

| platform | `secure` | `file` |
| --- | --- | --- |
| macOS | the Keychain, through `/usr/bin/security` (item `OpenRouter` or `InceptionLabs`, account `harness`) | `credentials.env` in the config folder, mode 0600 |
| Linux | the system keyring (Secret Service: GNOME Keyring, KWallet), only with the `keyring` extra installed and a store running | same |
| Windows | not supported | not supported |

On a headless Linux box with no keyring running, `file` is what you get. `harness config path`
shows the config folder; the credentials file sits next to the preferences file in it.

### Checking or changing the key

- Run `harness setup` again to replace a key or switch provider; the new key overwrites the old.
- The harness reads the secure store first, then the environment variable `OPENROUTER_API_KEY`
  or `INCEPTION_API_KEY`. The environment also covers a `.env` file and `credentials.env`, which
  load into it. So a stored key wins over an exported variable; to change the key, re-run
  `harness setup`. A harness that never ran `harness setup` still works with just the variable
  exported.
- On macOS, `security find-generic-password -s OpenRouter -a harness` (add `-s InceptionLabs`
  for Inception) shows whether an item exists without printing the key. Delete it with
  `security delete-generic-password -s OpenRouter -a harness`.
- A key in the file store is deleted by removing `credentials.env`.

## Check that it works

```bash
mkdir -p ~/agent-sandbox
harness run -w ~/agent-sandbox "write hello.py that prints hello, plus a pytest test for it"
```

If that finishes with a passing final check, you are set up. The workspace must be **outside**
any Python project; {doc}`safety` explains why. Next: {doc}`usage`.

## Working on the harness itself

From a clone, `uv sync` builds the environment and `uv run harness ...` runs it. To put the
command on your PATH so that editing `agent/` changes the installed executables at once:

```bash
uv tool install --editable .
```

`--editable` links the tool's environment to the source tree (dependency changes still need
`--force`). `uvx --from . harness-tui` also works but rebuilds every time. {doc}`development`
has the layout and the checks.
