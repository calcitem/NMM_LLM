# License

## NMM_LLM — Nine Men's Morris AI Engine

**Copyright (C) 2024–2026 Ben Brandwood**

> **Note on prior licensing:** Earlier versions of this repository displayed an MIT license badge in the README. That badge was informal and no MIT license file was ever formally deposited. The project is now formally and exclusively licensed under the GNU Affero General Public License, Version 3 (AGPL-3.0), as set out below. All original source code in this repository — Python, Rust, JavaScript, CSS, HTML, YAML, and shell scripts — is governed by AGPL-3.0.

---

## Primary License: GNU Affero General Public License v3.0

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU Affero General Public License** as published by the Free Software Foundation, either **version 3** of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but **WITHOUT ANY WARRANTY**; without even the implied warranty of **MERCHANTABILITY** or **FITNESS FOR A PARTICULAR PURPOSE**. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

### Summary of AGPL-3.0 obligations

- You may use, copy, modify, and distribute this software and derivative works.
- Any distribution of this software or a modified version **must** be accompanied by the complete corresponding source code under AGPL-3.0.
- If you run a **modified version of this software as a network service** (e.g. a hosted web game, analysis API, or online tool) and allow users to interact with it remotely, you **must** offer those users access to the corresponding source code under AGPL-3.0.
- All copyright notices and license texts must be preserved in all copies.

The full license text is available at: <https://www.gnu.org/licenses/agpl-3.0.txt>

---

## Third-Party Components

### 1. Malom Database — Nine Men's Morris Ultra-Strong Solution

Portions of the position/solution data accessed or indexed by this project originate from the **Malom** program and its associated endgame databases:

> **Malom: Ultra-Strong and Extended Solutions for Nine Men's Morris and Lasker Morris**
> Copyright (C) 2007–2023 Gábor E. Gevay and Gábor Danner
> University of Szeged, Hungary
> Web: <https://www.inf.u-szeged.hu/~danner/mills/>
> Source: <https://github.com/ggevay/malom>

Malom is distributed under the **GNU General Public License, Version 3 (GPL-3.0)**.

The GPL-3.0 is compatible with AGPL-3.0 and the Malom license terms are satisfied by this project's AGPL-3.0 licensing. The full GPL-3.0 license text is available at: <https://www.gnu.org/licenses/gpl-3.0.txt>

**Files in this repository that interface with Malom:**

| File | Role |
|------|------|
| `ai/malom_db.py` | MalomDB query wrapper |
| `tools/build_human_db.py` | Human DB builder — optionally annotates positions using Malom WDL/DTW |
| `tools/build_endgame_db.py` | Endgame DB builder — queries Malom for solution data |
| `tests/test_malom_db.py` | Integration tests for Malom query interface |

The Malom database files themselves (`.bin`, `.idx`, or similar binary files under paths such as `Std_DD_89adjusted/`) are **not** included in this repository and must be obtained separately from the Malom project under its GPL-3.0 license.

---

### 2. Python Runtime Dependencies

This project depends on third-party Python packages listed in `requirements.txt` and `requirements_learned_ai.txt`. These packages are not bundled in this repository and are obtained at install time via pip. Each package carries its own license. Key packages and their licenses are listed below for reference:

| Package | License |
|---------|---------|
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| Starlette | BSD-3-Clause |
| Jinja2 | BSD-3-Clause |
| Ollama (client) | MIT |
| ChromaDB | Apache-2.0 |
| NumPy | BSD-3-Clause |
| PyTorch (optional, `requirements_learned_ai.txt`) | BSD-3-Clause |
| python-docx | MIT |
| httpx | BSD-3-Clause |

These packages remain under their own respective licenses and are unaffected by this project's AGPL-3.0 license when used as libraries at runtime.

---

### 3. Rust Crate Dependencies

The native Rust extension (`native/nmm_core/`) uses the following crate:

| Crate | License |
|-------|---------|
| PyO3 | MIT / Apache-2.0 |

---

## Contact

Ben Brandwood
GitHub: <https://github.com/benmarkbrandwood-blip>

