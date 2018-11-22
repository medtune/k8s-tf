# k8s-tf
Messing with distributed TensorFlow and Kubernetes cluster.

---

## TODO
- use nfs as volume
- use persistentVolumes & persistentVolumeClaim for partitioning
- distributed training: mura
- distributed training: chestxray

###### Toughts
While writing training yaml files we understood why really [kubeflow]() exists... Configuring workers environnements and dns services can be automatically generated ! We'll continue messing with all this stuff in order to determine weither we should use kubeflow when its stable (version >= 1) or create our own generation tool.


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

#### Run cod training - CPU Only
`kubectl create -f deploy/train-cod`

#### Run cod training - GPU
`kubectl create -f deploy/train-cod-gpu`

#### Run tensorboard
`kubectl create -f deploy/tensorboard`


