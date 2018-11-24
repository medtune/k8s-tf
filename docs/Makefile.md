###### Create GKE cluster
`make create-cluster`

###### Configure kubectl context
`make kubectl`

###### Create namespace 
`kubectl create-namespace`

###### Configure default namespace as "medtune"
`make default-ns`

###### Create secrets for gcs 
`make create-secrets`

###### Prepare mnist data
`kubectl create -f job/prepare-mnist`

###### Run mnist training
`kubectl create -f deploy/train-mnist`

###### Prepare cod data
`kubectl create -f job/prepare-cod`

###### Run cod training - CPU Only
`kubectl create -f deploy/train-cod`

###### Run cod training - GPU
`kubectl create -f deploy/train-cod-gpu`

###### Run tensorboard
`kubectl create -f deploy/tensorboard`

