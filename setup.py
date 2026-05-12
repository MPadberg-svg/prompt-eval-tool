from setuptools import find_packages, setup

setup(
    name="prompt-eval-tool",
    version="0.1.0",
    description="CLI tool for evaluating LLM prompts",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["click>=8.1.0", "PyYAML>=6.0", "openai>=1.30.0,<3.0.0"],
    entry_points={"console_scripts": ["prompt-eval=prompt_eval.cli:cli"]},
    python_requires=">=3.10",
)
