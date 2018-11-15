# k8s-tf
Messing with distributed TensorFlow and Kubernetes cluster

---

###### Requirements:

Runing kubernetes cluster (v > 1.9)
Configured `kubectl`

#### Prepare container images
You can find ready to use images in [docker hub](https://hub.docker.com/r/medtune/k8s-tf)

`make build-images`
`make push-images`

#### Create namespace and persistentVolumeClaim

`kubectl create -f deploy/namespace.yaml`
`kubectl create -f deploy/data-pvc.yaml`

#### Create configMap
`kubectl create -f deploy/config-map.yaml`

#### Add secrets 
`kubectl create -f deploy/secrets.yaml`

#### Inspect persistentVolume
`kubectl create -f deploy/inspect-volume-job.yaml`

#### Run pre training job (mnist to tfrecord)
`kubectl create -f deploy/pre-train-job.yaml`

#### Run distributed training
