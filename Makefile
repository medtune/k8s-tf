build-images:
	# Build mnist to tfrecord job image
	docker build \
		-t medtune/k8s-tf:pre-train \
		-f build/pre-train.Dockerfile \
		jobs/convert-data

	# Build training job image
	docker build \
		-t medtune/k8s-tf:train \
		-f build/train.Dockerfile \
		jobs/train


push-images:
	# Push to docker hub
    # hub.docker.com/r/medtune/k8s-tf

	docker push medtune/k8s-tf:pre-train
	docker push medtune/k8s-tf:train


pull-images:
	# Pull images from docker hub

	docker pull medtune/k8s-tf:pre-train
	docker push medtune/k8s-tf:train