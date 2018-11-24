build-prepare-mnist:
	docker build \
		-t medtune/k8s-tf:prepare-mnist \
		-f build/images/prepare-mnist/Dockerfile \
		build/images/prepare-mnist

build-prepare-cod:
	docker build \
		-t medtune/k8s-tf:prepare-cod \
		-f build/images/prepare-cod/Dockerfile \
		build/images/prepare-cod


build-train-mnist:
	docker build \
		-t medtune/k8s-tf:train-mnist \
		-f build/images/train-mnist/Dockerfile \
		build/images/train-mnist

build-train-cod:
	docker build \
		-t medtune/k8s-tf:train-cod-cpu \
		-f build/images/train-cod/Dockerfile \
		build/images/train-cod

build-train-cod-cpu: build-train-cod

build-train-cod-gpu:
	docker build \
		-t medtune/k8s-tf:train-cod-gpu \
		-f build/images/train-cod/Dockerfile.gpu \
		build/images/train-cod

build-images: build-train-cod \
	build-train-mnist \
	build-prepare-cod \
	build-prepare-mnist \
	build-train-cod-gpu


push-images:
	# Push to docker hub
    # hub.docker.com/r/medtune/k8s-tf
	docker push medtune/k8s-tf:train-mnist
	docker push medtune/k8s-tf:prepare-mnist
	docker push medtune/k8s-tf:prepare-cod
	docker push medtune/k8s-tf:train-cod-cpu
	docker push medtune/k8s-tf:train-cod-gpu

pull-images:
	# Pull images from docker hub
	docker pull medtune/k8s-tf:prepare-mnist
	docker push medtune/k8s-tf:train-mnist
	docker pull medtune/k8s-tf:prepare-cod
	docker push medtune/k8s-tf:train-cod-cpu
	docker push medtune/k8s-tf:train-cod-gpu

create-namespace:
	kubectl create -f meta/namespace.yaml

prepare-mnist:
	kubectl create -f k8s/jobs/prepare-mnist

prepare-cod:
	kubectl create -f k8s/jobs/prepare-cod

train-mnist:
	kubectl create -f k8s/deployments/train-mnist

train-cod:
	kubectl create -f k8s/deployments/train-cod

train-cod-gpu:
	kubectl create -f k8s/deployments/train-cod-gpu

default-ns:
	kubectl config \
		set-context $(shell kubectl config current-context) \
		--namespace="medtune"

create-secrets:
	kubectl create secret generic gcs-creds \
		-n medtune \
		--from-file=./secrets/mdtn.json

kubectl:
	gcloud container clusters get-credentials test-vcluster \
		--zone europe-west1-b \
		--project medtune-europe

create-cluster:
	gcloud container clusters create test-vcluster \
      --zone europe-west1-b \
      --num-nodes 1 \
      --cluster-version 1.11 \
      --disk-size 100 \
      --machine-type n1-standard-8

create-gpu-cluster:
	gcloud container clusters create test-vcluster \
      --zone europe-west1-b \
      --num-nodes 1 \
      --cluster-version 1.11 \
      --disk-size 100 \
      --machine-type n1-standard-8

delete-cluster:
	gcloud container clusters delete test-vcluster

