FROM hub.cc-asp.fraunhofer.de/medicalmultitaskmodeling/mmm-stack:1.0.0

WORKDIR /app

COPY poetry.lock pyproject.toml README.md ./
COPY mmm ./mmm

RUN poetry install

COPY api ./api

CMD uvicorn api.api:app --port 9504 --host 0.0.0.0
