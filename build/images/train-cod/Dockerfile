FROM tensorflow/tensorflow:1.12.0-py3 


RUN apt update && apt install -y git

WORKDIR /code

RUN git clone -b stats_distribute https://github.com/medtune/Models-implementation .

# Setup research & slim
RUN git clone -b master https://github.com/tensorflow/models

RUN cp -r models/research .

RUN cd research && \
    python3 setup.py install
RUN cd research/slim && \
    python3 setup.py install

# Add all project files and dirs
ADD . .

# Entrypoint will get executed after starting the container
ENTRYPOINT [ "python3", "estimator.py" ]