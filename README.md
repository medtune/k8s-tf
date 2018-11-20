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

#### Create GKE cluster
`make create-cluster`

#### Configure kubectl context
`make kubectl`

#### Create namespace 
`kubectl create-namespace`

#### Configure default namespace as "medtune"
`make default-ns`

#### Create secrets for gcs 
`make create-secrets`

#### Prepare mnist data
`kubectl create -f job/prepare-mnist`

#### Run mnist training
`kubectl create -f deploy/train-mnist`

#### Prepare cod data
`kubectl create -f job/prepare-cod`

#### Run cod training
`kubectl create -f deploy/train-cod`

#### Run tensorboard
`kubectl create -f deploy/tensorboard`


