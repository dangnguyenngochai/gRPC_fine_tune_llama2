FROM nvidia/cuda

RUN apt update \
 pip install -r requirement.txt

RUN apt-get update
RUN apt-get install python3 python3-pip -y

COPY ./requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt


