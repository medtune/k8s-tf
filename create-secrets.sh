#!/bin/bash

kubectl create secret generic gcs-creds -n medtune --from-file=./secrets/mdtn.json