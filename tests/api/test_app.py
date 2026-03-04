from pathlib import Path
import pytest
from typer.testing import CliRunner

from mmm.app import app

runner = CliRunner()


def test_status():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "importable!" in result.stdout


def test_create_demo_data():
    from mmm.api.PetTestData import PetTestData

    if not PetTestData.data_exists():
        pytest.skip("PetTestData data does not exist")
    with runner.isolated_filesystem() as tempdir:
        tempfolder = Path(tempdir)
        result = runner.invoke(app, ["create-demo-data", str(tempfolder), "--num-cases", "2", "-y"])
        assert result.exit_code == 0
        assert tempfolder.exists()
        assert len(list(tempfolder.iterdir())) == 2
