FROM google/cloud-sdk:latest

WORKDIR /data

ENV GOOGLE_APPLICATION_CREDENTIALS /data/creds.json

RUN mkdir cod

ADD download.sh .

RUN chmod +x download.sh

ENTRYPOINT ["gsutil", "cp", "-r", "gs://medtune/datasets/catvsdog", "./cod" ]