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


build-train-mura-cpu:
	docker build --no-cache \
		-t medtune/k8s-tf:train-mura-cpu \
		-f build/images/train-mura/Dockerfile \
		build/images/train-mura
		
build-train-mura-gpu:
	docker build --no-cache \
		-t medtune/k8s-tf:train-mura-gpu \
		-f build/images/train-mura/Dockerfile.gpu \
		build/images/train-mura

build-train-mura-gpu-v2:
	docker build --no-cache \
		-t medtune/k8s-tf:train-mura-gpu-v2 \
		-f build/images/train-mura/v2/Dockerfile.gpu \
		build/images/train-mura/v2

create-namespace:
	kubectl create -f k8s/namespace.yaml

prepare-mnist:
	kubectl create -f k8s/jobs/prepare-mnist

prepare-cod:
	kubectl create -f k8s/jobs/prepare-cod

prepare-mura:
	kubectl create -f k8s/jobs/prepare-mura
	kubectl create -f k8s/jobs/dl-mobilenet-v2

train-mnist:
	kubectl create -f k8s/deployments/train-mnist

train-cod:
	kubectl create -f k8s/deployments/train-cod

train-cod-cpu: train-cod

train-cod-gpu:
	kubectl create -f k8s/deployments/train-cod-gpu

train-mura-gpu:
	kubectl create -f k8s/deployments/train-mura-gpu

train-mura-gpu-v2:
	kubectl create -f k8s/deployments/train-mura-gpu-v2

force-stop:
	kubectl delete deployment --all
	kubectl delete service --all

default-ns:
	kubectl config \
		set-context $(shell kubectl config current-context) \
		--namespace="medtune"

create-secrets:
	kubectl create secret generic gcs-creds \
		-n medtune \
		--from-file=./secrets/mdtn.json

	kubectl create secret generic aws-creds \
		-n medtune \
		--from-file=./secrets/credentials

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

nvidia-ds:
	kubectl \
		apply \
		-f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml


nfs-pv:
	kubectl create -f \
		k8s/nfs/

init-pv:
	kubectl create -f \
		k8s/nfs/init/

dl-mobilenet:
	kubectl create -f \
		k8s/jobs/dl-mobilenet-v2

delete-cluster:
	gcloud container clusters delete test-vcluster

build-images: build-prepare-mnist \
	build-train-mnist \
	build-train-cod-cpu \
	build-train-cod-gpu \
	build-train-mura-gpu-v2

push-images:
	# Push to docker hub
    # hub.docker.com/r/medtune/k8s-tf
	docker push medtune/k8s-tf:train-mnist
	docker push medtune/k8s-tf:prepare-mnist
	#docker push medtune/k8s-tf:prepare-cod
	#docker push medtune/k8s-tf:train-cod-cpu
	docker push medtune/k8s-tf:train-cod-gpu
	#docker push medtune/k8s-tf:train-mura-cpu
	#docker push medtune/k8s-tf:train-mura-gpu
	docker push medtune/k8s-tf:train-mura-gpu-v2

pull-images:
	# Pull images from docker hub
	docker pull medtune/k8s-tf:prepare-mnist
	docker pull medtune/k8s-tf:train-mnist
	#docker pull medtune/k8s-tf:prepare-cod
	docker pull medtune/k8s-tf:train-cod-cpu
	docker pull medtune/k8s-tf:train-cod-gpu
	#docker pull medtune/k8s-tf:train-mura-cpu
	docker pull medtune/k8s-tf:train-mura-gpu
	docker pull medtune/k8s-tf:train-mura-gpu-v2
