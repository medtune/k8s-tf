FROM tensorflow/tensorflow:1.12.0-py3

RUN apt update -y && \
    apt upgrade -y

ADD mnist2tfrecord.py .

ENTRYPOINT [ "python3", "mnist2tfrecord.py" ]