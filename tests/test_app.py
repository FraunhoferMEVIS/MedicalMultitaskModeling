import os
from typer.testing import CliRunner

from mmm.app import app

runner = CliRunner()


def test_status():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "importable!" in result.stdout
