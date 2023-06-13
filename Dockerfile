FROM pytorch/pytorch
WORKDIR /app

# install opencv dependencies
RUN apt-get update && apt-get install libgl1 libglib2.0-0 -y

COPY requirements_docker.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
CMD ["python3", "-u", "./restapi.py", "--port", "5000"]