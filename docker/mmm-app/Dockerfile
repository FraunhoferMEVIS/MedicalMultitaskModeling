ARG MMMVERSION=setme

FROM hub.cc-asp.fraunhofer.de/medicalmultitaskmodeling/mmm-stack:${MMMVERSION}

WORKDIR /app

COPY poetry.lock pyproject.toml README.md ./
COPY mmm ./mmm

CMD uvicorn mmm.api:app --port 9504 --host 0.0.0.0
