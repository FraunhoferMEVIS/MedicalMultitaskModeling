try:
    import fastapi

    api_extra_installed = True
except ImportError:
    api_extra_installed = False

from mmm.inference_app import build_app, APISettings

if api_extra_installed:
    app = build_app(APISettings())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
