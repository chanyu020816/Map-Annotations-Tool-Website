FROM python:3.10

WORKDIR /website

RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && apt-get install -y gcc g++ \
    && apt-get clean

COPY . /website/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "app.py"]

# docker build -t map_ano .
# docker run -p 8000:8000 --rm --mount src="$(pwd)",target=/website,type=bind map_ano
# docker run -p 8000:8000 --rm --mount src="$(pwd)",target=/website,type=bind map_ano python app.py
# docker compose up -d