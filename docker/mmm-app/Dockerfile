ARG MMMVERSION=setme

FROM hub.cc-asp.fraunhofer.de/medicalmultitaskmodeling/mmm-stack:${MMMVERSION}

WORKDIR /app

COPY poetry.lock pyproject.toml README.md ./
COPY mmm ./mmm

RUN poetry install -E all

CMD mmm inference-app --port 9504
