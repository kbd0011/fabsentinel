"""Smoke tests for the Typer CLI."""
from typer.testing import CliRunner

from fabsentinel.cli import app


def test_help_renders():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "scorecard" in result.stdout


def test_ingest_help():
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
