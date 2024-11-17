FROM python:3.11-slim

RUN pip --no-cache-dir install pipenv

WORKDIR /app                                                                

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy --system && \
    rm -rf /root/.cache

COPY ["src/predict.py", "src/"]
COPY ["models/", "models/"]

WORKDIR /app/src

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]