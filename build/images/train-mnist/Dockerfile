FROM tensorflow/tensorflow:1.12.0-py3

RUN apt update -y && \
    apt upgrade -y

ADD distributed_training.py .

ENTRYPOINT [ "python3", "distributed_training.py" ]