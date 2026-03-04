from pathlib import Path
from zipfile import ZipFile
from mmm.logging.ZipLog import ZipLog


def test_ziplog(tmp_path: Path):
    # Adding files to the log
    with (log := ZipLog()).add_files() as logdir:
        logdir.joinpath("log.txt").write_text("Top-level log file.")
        logdir.joinpath("subdir").mkdir()
        logdir.joinpath("subdir/log.txt").write_text("Subdirectory log file.")
        assert logdir.exists()
    assert not logdir.exists()

    # When exiting the context, the zip file should be uploaded automatically
    assert log.upload_path.exists()

    # Download the zip file to a local path for inspection
    local_zip_path = tmp_path / "downloaded_log.zip"
    local_zip_path.write_bytes(log.upload_path.upath().read_bytes())

    # Inspect the contents of the zip file
    with ZipFile(local_zip_path, "r") as zip_file:
        namelist = zip_file.namelist()
        assert "log.txt" in namelist
        assert "subdir/log.txt" in namelist
