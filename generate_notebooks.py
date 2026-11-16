#!/usr/bin/env python3
"""
Generate Jupyter Notebooks (.ipynb) for all 30 Capstone Chapters.
Author: Kalyanjit Hatibaruah
Copyright: © 2026 Flugelsoft Labs
"""

import json
import re
from pathlib import Path

def markdown_to_notebook(md_content: str, title: str, slug: str) -> dict:
    cells = []

    # 1. Header cell
    header_text = [
        f"# {title}\n",
        "**Data Science & Machine Learning for Physics: 30 Capstone Projects Across Industries**  \n",
        "*Author: Kalyanjit Hatibaruah* | *Publisher: Flugelsoft Labs (2026)*  \n",
        f"*Online Static Book Chapter: https://flugelsoft-01.github.io/physics-ml-book/chapters/{slug}.html*\n",
        "\n---"
    ]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": header_text
    })

    # 2. Environment Setup cell
    setup_code = [
        "# Colab / Cloud GPU Environment Setup\n",
        "!pip install -q numpy scipy torch scikit-learn matplotlib plotly\n",
        "!mkdir -p shared_utils\n",
        "!wget -q https://raw.githubusercontent.com/flugelsoft-01/physics-ml-book/main/shared_utils/physics_helpers.py -O shared_utils/physics_helpers.py\n",
        "!wget -q https://raw.githubusercontent.com/flugelsoft-01/physics-ml-book/main/shared_utils/plot_helpers.py -O shared_utils/plot_helpers.py\n",
        "print('✓ Physics-ML environment initialized!')"
    ]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": setup_code
    })

    # Split md_content into code blocks and markdown sections
    # Pattern to match ```python ... ```
    pattern = re.compile(r"```python(.*?)```", re.DOTALL)
    last_end = 0

    for match in pattern.finditer(md_content):
        # Text before code block
        md_part = md_content[last_end:match.start()].strip()
        if md_part:
            # Clean up markdown text for notebook readability
            lines = [line + "\n" for line in md_part.split("\n")]
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": lines
            })

        # Python Code block
        code_part = match.group(1).strip()
        if code_part:
            code_lines = [line + "\n" for line in code_part.split("\n")]
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_lines
            })

        last_end = match.end()

    # Remaining markdown
    remaining_md = md_content[last_end:].strip()
    if remaining_md:
        lines = [line + "\n" for line in remaining_md.split("\n")]
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    nb = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": f"{title} - Physics-ML Capstone",
                "provenance": [],
                "collapsed_sections": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    return nb


def generate_all_notebooks():
    root = Path(__file__).resolve().parent
    chapters_dir = root / "chapters"
    notebooks_dir = root / "notebooks"
    dist_notebooks_dir = root / "dist" / "notebooks"

    notebooks_dir.mkdir(parents=True, exist_ok=True)
    dist_notebooks_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for md_file in sorted(chapters_dir.glob("part*/*.md")):
        slug = md_file.stem
        raw_md = md_file.read_text(encoding="utf-8")

        # Extract title
        title_match = re.search(r"^#\s+(.+)$", raw_md, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else slug.replace("_", " ").title()

        # Generate notebook dict
        nb_data = markdown_to_notebook(raw_md, title, slug)

        # Write to notebooks/ and dist/notebooks/
        out_file = notebooks_dir / f"{slug}.ipynb"
        dist_out_file = dist_notebooks_dir / f"{slug}.ipynb"

        nb_json = json.dumps(nb_data, indent=2)
        out_file.write_text(nb_json, encoding="utf-8")
        dist_out_file.write_text(nb_json, encoding="utf-8")
        count += 1
        print(f"  ✓ Generated: notebooks/{slug}.ipynb")

    print(f"\nSuccessfully generated {count} Jupyter notebooks in notebooks/ and dist/notebooks/!")

if __name__ == "__main__":
    generate_all_notebooks()
