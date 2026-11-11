# Security Policy

## Supported Versions

Security fixes are applied to the latest release of the `deepguard` PyPI package and to the `main` branch.

| Version | Supported |
| :--- | :---: |
| 1.1.x (latest) | ✅ |
| < 1.1 | ❌ |

## Reporting a Vulnerability

**Please do not open a public issue, pull request, or discussion for security problems.**

Report vulnerabilities privately through GitHub:
[**Report a vulnerability**](https://github.com/HanMoonSub/DeepGuard/security/advisories/new)
(Security tab → *Report a vulnerability*).

If you cannot use GitHub's private reporting, email **seoyunje2001@gmail.com** instead.

Please include:

- The affected component (`deepguard` package, `inference`/`explainability`, `App/` backend, `client/` frontend, or Docker images)
- The version or commit you tested
- Steps to reproduce, or a proof of concept
- The impact you expect (e.g. data exposure, remote code execution, session hijacking)

## What to Expect

- We aim to acknowledge your report within **7 days**.
- We will keep you updated while we investigate and work on a fix.
- Once a fix is released, we will publish a GitHub Security Advisory and credit you unless you ask us not to.

DeepGuard is maintained by a small student team, so response times may vary. Thank you for helping keep the project safe.

## Scope

In scope:

- The `deepguard`, `inference`, `explainability`, and `preprocess` Python packages
- The FastAPI backend (`App/`), including session handling and file uploads
- The React frontend (`client/`)
- Docker images and `docker-compose.yml` published from this repository

Out of scope:

- Vulnerabilities in third-party dependencies that are already publicly known (please report them upstream; Dependabot tracks updates here)
- Model evasion or adversarial examples that make a deepfake go undetected — these are research limitations, not security vulnerabilities, and can be reported as regular issues
