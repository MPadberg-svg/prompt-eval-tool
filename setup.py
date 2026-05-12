from pathlib import Path

from setuptools import find_packages, setup

README_PATH = Path(__file__).parent / "README.md"

setup(
    name="prompt-eval-tool",
    version="0.1.0",
    description="CLI tool for evaluating LLM prompts",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Matteo Padberg",
    author_email="MPADBERG@my.lonestar.edu",
    url="https://github.com/MPadberg-svg/prompt-eval-tool",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click>=8.1.0",
        "PyYAML>=6.0",
        "openai>=1.30.0,<3.0.0",
        "pandas>=2.0.0",
        "langdetect>=1.0.9",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "pytest>=8.0.0",
    ],
    entry_points={"console_scripts": ["prompt-eval=prompt_eval.cli:cli"]},
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
