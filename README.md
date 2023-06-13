# Ludus AI Referee

A proof of concept made for Ludus Alliance by project group 7 in 2023.

![image](https://github.com/JurekInholland/LudusApi/assets/42969112/8f53acb5-e01f-4179-bdfa-f0e05efc5d02)


## Getting started
```bash
#clone this repo
git pull https://github.com/JurekInholland/LudusApi.git
cd LudusApi

# create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# install requirements
 pip install -r requirements.txt

# run application
python restapi.py --port 5000
```

### Docker
```bash
docker build -f Dockerfile -t ludus-api .
docker run --rm --gpus all -p 5000:5000 ludus-api
```

### Docker compose
```bash
docker compose up --build
```

View http://127.0.0.1:5000/ in a browser or send POST requests to http://127.0.0.1:5000/api/analyse
