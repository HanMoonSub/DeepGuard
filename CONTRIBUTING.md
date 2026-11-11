# Contributing to DeepGuard

Thanks for your interest in DeepGuard! Bug reports, documentation fixes, and code contributions are all welcome.

## Before You Start

- **Read [`AI_POLICY.md`](./AI_POLICY.md).** If you use AI tools, disclose AI-generated text and make sure you understand every line of code you submit.
- **Search existing issues and pull requests** before opening a new one, to avoid duplicates.
- For larger changes (new models, new datasets, API changes), please open an issue first so we can discuss the approach.
- Security problems should **not** be reported as issues — see [`SECURITY.md`](./.github/SECURITY.md).

## Development Setup

```bash
git clone https://github.com/HanMoonSub/DeepGuard.git
cd DeepGuard
pip install -r requirements.txt
```

Python 3.11 is recommended (the package supports Python >= 3.10).

To run the full web app, see the [Docker Quick Start](./README.md#-docker-quick-start) in the README.

## Branches

- `main` — released code
- `develop` — integration branch for ongoing work

Create your branch from `develop` and open your pull request **into `develop`**.

## Before Opening a Pull Request

Run these from the repository root and make sure they pass:

```bash
ruff check .
ruff format --check .
pytest --tb=short -q
```

Frontend changes (`client/`):

```bash
cd client
npm test
npm run build
```

## Pull Request Guidelines

- Keep each pull request focused on one change.
- Describe what changed and why, and link the related issue.
- Commit messages follow [gitmoji](https://gitmoji.dev/) style, e.g. `🐛 Fix face crop margin for portrait videos`.
- Write code comments and docstrings in English.
- When adding dual-branch model parameters, keep the `l_` (low-level branch) / `h_` (high-level branch) prefix convention.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).
