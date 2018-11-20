build-prepare-mnist:
	# Build mnist to tfrecord job image
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
	# Build training job image
	docker build \
		-t medtune/k8s-tf:train-mnist \
		-f build/images/train-mnist/Dockerfile \
		build/images/train-mnist

build-train-cod:
	docker build \
		-t medtune/k8s-tf:train-cod \
		-f build/images/train-cod/Dockerfile \
		build/images/train-cod

build-train-cod-gpu:
	docker build \
		-t medtune/k8s-tf:train-cod-gpu \
		-f build/images/train-cod/Dockerfile.gpu \
		build/images/train-cod

build-images: build-train-cod \
	build-train-mnist \
	build-prepare-cod \
	build-prepare-mnist


push-images:
	# Push to docker hub
    # hub.docker.com/r/medtune/k8s-tf
	docker push medtune/k8s-tf:train-mnist
	docker push medtune/k8s-tf:train-cod
	docker push medtune/k8s-tf:prepare-mnist
	docker push medtune/k8s-tf:prepare-cod

pull-images:
	# Pull images from docker hub
	docker pull medtune/k8s-tf:prepare-mnist
	docker push medtune/k8s-tf:train-mnist
	docker pull medtune/k8s-tf:prepare-cod
	docker push medtune/k8s-tf:train-cod

create-namespace:
	kubectl create -f meta/namespace.yaml

prepare-mnist:
	kubectl create -f job/prepare-mnist

prepare-cod:
	kubectl create -f job/prepare-cod

train-mnist:
	kubectl create -f deploy/train-mnist

train-cod:
	kubectl create -f deploy/train-cod

train-cod-gpu:
	kubectl create -f deploy/train-cod-gpu

default-ns:
	bash scripts/default-ns.sh medtune

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

delete-cluster:
	gcloud container clusters delete test-vcluster