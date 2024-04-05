# Setup

```sh
docker build -t map_ano
docker run -p 8000:8000 --rm --mount src="$(pwd)",target=/website,type=bind map_ano
```