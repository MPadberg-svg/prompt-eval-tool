import re

from click.testing import CliRunner

from prompt_eval.cli import cli


def test_help_lists_three_commands():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    command_lines = re.findall(r"^\s{2}(evaluate|export|list-models)\s", result.output, re.MULTILINE)
    assert sorted(command_lines) == ["evaluate", "export", "list-models"]
