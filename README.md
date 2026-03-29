# Deep Learning VT2026 at Jönköping Univeristy
Personal repository for all things Exercises, Assignments and Projects of the Deep Learning VT2026 course at Jönköping University.
## Handling Dependencies
I recommend handling the Python environment and dependencies with uv.

uv is an extremely fast, Rust-based Python package manager and project manager designed to replace and unify tools like pip, pip-tools, pipx, poetry, and virtualenv. It is developed by Astral and is 10–100x faster than pip for installing packages, offering instant virtual environment creation and dependency resolution.

Please install uv: https://docs.astral.sh/uv/getting-started/installation/

After installing uv and cloning the repository, all you have to do is run uv sync in your project root. Done!

If you need to add any dependencies, pleaso do so by running uv add package-name.