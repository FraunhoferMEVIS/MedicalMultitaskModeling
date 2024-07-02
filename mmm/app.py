import os
import typer

app = typer.Typer()


@app.command()
def status():
    from mmm.settings import mtl_settings

    typer.secho("MMM Settings", fg=typer.colors.BRIGHT_BLUE)
    typer.secho(mtl_settings.model_dump_json(indent=2))

    # Test if mmm is importable
    typer.secho("Testing if mmm is importable...", fg=typer.colors.BRIGHT_BLUE)
    from mmm.interactive import blocks, configs, data, pipes, tasks, training, utils

    typer.secho("mmm is importable!", fg=typer.colors.GREEN)


@app.command()
def inference_app(host: str = "0.0.0.0", port: int = 8000):
    try:
        import uvicorn
        import fastapi
    except ImportError:
        typer.secho("Please install the extra dependencies 'api' of medicalmultitaskmodeling", fg=typer.colors.RED)
        return
    from mmm.inference_app import build_app, APISettings

    typer.secho(
        f"Use environment variables to configure the app: {APISettings.Config.env_prefix}", fg=typer.colors.BRIGHT_BLUE
    )
    if "MTLAPI_modules_path" not in os.environ:
        typer.secho(
            f"For example: `MTLAPI_modules_path=/path/to/model.pt.zip mmm inference-app` to use local model",
            fg=typer.colors.BRIGHT_BLUE,
        )
    api_settings = APISettings()
    app = build_app(api_settings)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    app()
