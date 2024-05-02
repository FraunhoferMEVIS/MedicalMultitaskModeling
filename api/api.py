from mmm.inference_api import build_app, APISettings

app = build_app(APISettings())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
