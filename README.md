# k8s-tf
Messing with distributed TensorFlow and Kubernetes cluster.

[![CircleCI](https://circleci.com/gh/medtune/k8s-tf/tree/master.svg?style=svg)](https://circleci.com/gh/medtune/k8s-tf/tree/master)

---

## TODO
- distributed training: chestxray

###### Toughts
While writing training yaml files we understood why really [kubeflow](https://github.com/kubeflow/kubeflow) exists... Configuring workers environnements and dns services is not complicated but time consuming. It can be automatically generated ! We'll continue messing with all this stuff in order to determine weither we should use kubeflow when its stable (version >= 1) or create our own generation tool.


#### Requirements:
- Installed and configured `gcloud` (verify that you have enough quotas for using [GKE](https://cloud.google.com/kubernetes-engine/))
- Installed `kubectl`

#### Documentation

See [./docs](./docs)