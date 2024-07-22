+++
title = "Kubernetes Up and Running"
description = "Comprehensive overview of Kubernetes"
+++
# 1 Introduction


# 2 Creating and Running Applications
### Container Images
-  A container image is a binary package that encapsulates all of the files necessary to run a program inside of an OS container.
- Most popular format is docker
- Container images are constructed with a series of filesystem layers, where each layer inherits and modifies the layers that came before it
- 2 categories of images
    - system: mimics virtual machine and runs full boot process.  Seen as poor practice.
    - application: runs a single program

### Building Application Images With Docker
- Dockerfile
```Dockerfile
# Start from a Node.js 16 (LTS) image 
FROM node:16

# Specify the directory inside the image in which all commands will run 
WORKDIR /usr/src/app

# Copy package files and install dependencies 
COPY package*.json ./
RUN npm install
RUN npm install express

# Copy all of the app files into the image 
COPY . .

# The default command to run when starting the container 
CMD [ "npm", "start" ]
```

- Commands
```
docker build -t simple-node .
docker run --rm -p 3000:3000 simple-node**
```

- Building image efficiently:
    - order your layers from least likely to change to most likely to change in order to optimize the image size for pushing and pulling. This is why, in Example 2-4, we copy the package*.json files and install dependencies before copying the rest of the program files

- ==Do not mix secrets with images==
    - Include as less content as possible


### Multistage Image Builds
- Build image efficiently by including only what is necessary
    - Ex: include only the necessary binary and none of the development
- Docker introduced _multistage builds_. With multistage builds, rather than producing a single image, a Docker file can actually produce multiple images. Each image is considered a stage. ==Artifacts can be copied from preceding stages to the current stage==
    - Building a container image using multistage builds can reduce your final container image size by hundreds of megabytes and thus dramatically speed up your deployment times

- Example: Single Image; inefficient
```Dockerfile
FROM golang:1.17-alpine

# Install Node and NPM
RUN apk update && apk upgrade && apk add --no-cache git nodejs bash npm

# Get dependencies for Go part of build
RUN go get -u github.com/jteeuwen/go-bindata/...
RUN go get github.com/tools/godep
RUN go get github.com/kubernetes-up-and-running/kuard

WORKDIR /go/src/github.com/kubernetes-up-and-running/kuard

# Copy all sources in
COPY . .

# This is a set of variables that the build script expects
ENV VERBOSE=0
ENV PKG=github.com/kubernetes-up-and-running/kuard
ENV ARCH=amd64
ENV VERSION=test

# Do the build. This script is part of incoming sources.
RUN build/build.sh

CMD [ "/go/bin/kuard" ]
```


- Ex: Multi-stage Build
    - first image is built as build
    - second stage uses the build image
```Dockerfile
# STAGE 1: Build
FROM golang:1.17-alpine AS build

# Install Node and NPM
RUN apk update && apk upgrade && apk add --no-cache git nodejs bash npm

# Get dependencies for Go part of build
RUN go get -u github.com/jteeuwen/go-bindata/...
RUN go get github.com/tools/godep

WORKDIR /go/src/github.com/kubernetes-up-and-running/kuard

# Copy all sources in
COPY . .

# This is a set of variables that the build script expects
ENV VERBOSE=0
ENV PKG=github.com/kubernetes-up-and-running/kuard
ENV ARCH=amd64
ENV VERSION=test

# Do the build. Script is part of incoming sources.
RUN build/build.sh

# STAGE 2: Deployment
FROM alpine

USER nobody:nobody
COPY --from=build /go/bin/kuard /kuard

CMD [ "/kuard" ]
```

### Deploying to Artifactory
- Tags represent a variant of an image, like version
    - blue is the variant
```
docker login ...
docker tag kuard gcr.io/kuar-demo/kuard-amd64:blue
docker push gcr.io/kuar-demo/kuard-amd64:blue
```


### Container Runtime Interface (CRI)
- K8s defines the application deployment; CRI implements the definition for a specific OS.
    - For linux, CRI configures cgroups and namespaces
    - Vendor implementation
        - containerd-cri by Docker
        - cri-o by Red Had


### Running Containers With Dockers
```terminal
docker r**run -d --name kuard \
  --publish 8080:8080 \
  --memory 200m \
  --memory-swap 1G \
  --cpu-shares 1024 \
  gcr.io/kuar-demo/kuard-amd64:blue
```
- --publish (-p ) is port forward
- -d means run in the background as a daemon
- --name gives container a friendly name
- limit the resources

### Cleanup
- Stop and remove container
```terminal
docker stop kuard
docker rm kuard
```


- Remove images
``` terminal
docker rmi <_tag-name_>
docker rmi <_image-id_>
```

# 3 Deploying K8s Cluster
### 3.1 Installing K8s as a Public Cloud Provider

```terminal

# Google
gcloud config set compute/zone us-west1-a
gcloud container clusters create kuar-cluster --num-nodes=3
gcloud container clusters get-credentials kuar-cluster

# AWS
eksctl create cluster
**eksctl create cluster --help**

```

### 3.2 Installing K8s Locally With MiniKube
```terminal
minikube start
minikube stop
minikube delete
```

### 3.3 Running K8s In Docker
- Use Docker containers to simulate multiple K8s nodes instead of running on virtual machines
    - [K8s in Docker KIND](https://kind.sigs.k8s.io/)

```terminal
kind create cluster --wait 5m
export KUBECONFIG="$(kind get kubeconfig-path)"
kubectl cluster-info
kind delete cluster
```


### 3. 4 K8s Client
- kubectl (ktl)

```terminal
# Verify health of cluster components: scheduler, controller-manager, etcd0
kubectl get componentstatuses

# List the cluster nodes
kubectl get nodes

# Describe a specific node
ktl describe nodes kube1

```

### 3.5 Cluster Components
- Runs in the kube-system namespace
##### K8s Proxy
- K8s Proxy is responsible to route network traffic to load balanced services in cluster; the proxy runs in every node of the cluster
- Is executed as an api object Daemonset (learn in Ch 11)
```
$ kubectl get daemonSets --namespace=kube-system kube-proxy
NAME         DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR
kube-proxy   5         5         5       5            5           ...   45d
```

##### K8s DNS
- DNS server provides naming and discovery for the services
- Is executed as an api object deployment
- cmd
```
$ kubectl get deployments --namespace=kube-system core-dns
NAME       DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
core-dns   1         1         1            1           45d
```

##### K8s UI



# 4 Common Kubectl Commands
### 4.1 Namespaces
- namespaces are used to organize objects in cluster; ~ a folder
- ==default== is the namespace by default
    - use --namespace to specify other name spaces
    - use --all-namespace for all namespaces
### 4.2 Contexts
- To update/change what the default namespace, use context
    - Value is stored in $HOME/.kube/config

- 2 step procedure
```terminal
# Define the Context
kubectl config set-context my-context --namespace=mystuff

# Use the context
kubectl config use-context my-context

```



### 4.3 Viewing K8s API Objects
- K8s objects is represented as a RESTFUL resource
    - ie: https://your-k8s.com/api/v1/namespaces/default/pods/my-pod

- Get : For basic information
  ```
  kubectl get <resource-name> <obj-name>

  # more details
  kubectl get <resource-name> <obj-name> -o wide

  # specify output format (json or yaml)
  kubectl get <resource-name> <obj-name> -o json

  # skip header, to make it easier to use awk
  kubectl get <resource-name> <obj-name> --no-headers

  # pull specific information, using jsonpath query language
  kubectl get pods my-pod -o jsonpath --template={.status.podIP}

  # View multiple resources at once
  kubectl get pods,services
  
  ```

    - Describe for more information
``` terminal
	kubectl describe <resource-name> <obj-name>

```


- Explain: To see supported fields of a resource type
    - Ex: ktl explain pods



### 4.4 Create, Update, Destroy K8s Objects
- K8s objects represented as json or yaml files
- Via apply
```
	ktl apply -f obj.yaml

	# Prints the object without sending to server
	ktl apply -f obj.yaml. --dry-run 

	# apply records a history of modification; one can backtrack to one of them with: edit-last-applied, set-last-applied, view-last-applied
	ktl apply -f myobj.yaml view-last-applied
```

- Via edit: interactive edit of an active k8s object
```
	ktl edit <resource-name> <obj-name>
```


- Delete
``` terminal

	ktl delete -f obj.yaml
	ktl delete <resource-name> <obj-name>
```

### 4.5 Labeling and Annotating Objects
- Labels and annotations are tags for objects. Chap 6 discusses the difference between the two
- Commands
```
	ktl label pods bar color=red

	# To overwrite an existing label
	ktl label pods bar color=blue --overwrite

	# Remove the color label from pod
	ktl label podsbar color-
	
	
```


### 4.6 Debugging Commands
```
	ktl logs pod-name
	
	# A pod can have multiple containers
	ktl log pod-name -c 

	# stream a log
	ktl log -f pod-name

	# Execute command in a running container
	ktl exec -it POD-NAME -- bash

	# If pod has no bash, one can attach
	ktl attach -it POD-NAME

	# Copy a file from running container to local machine
	ktl cp <pod-name>:</path/to/remote/file> </path/to/local/file>

	# Port forwarding; Forwards traffic from the local machine on port 8080 to the remote container on port 80
	ktl port-forward POD-NAME 8080:80

	# See events in hte given namespace
	ktl get events

	# Stream the events
	ktl get events --watch

	# Display CPU memory; may need to include namespace
	ktl top pods

``` 



### 4.7 Cluster Management
- Cordon a node: prevent future pods from being scheduled on a particular node in cluster
``` terminal
	ktl cordon
	ktl drain
```

-


### 4.8 Command Autocompletion
```
# macOS
$ brew install bash-completion

# CentOS/Red Hat
$ yum install bash-completion

# Debian/Ubuntu
$ apt-get install bash-completion

brew install bash-completion
source <(kubectl completion bash)
echo "source <(kubectl completion bash)" >> ${HOME}/.bashrc

```


### 4.9 Alternative Way of Viewing Cluster
- Via plugins
    - IntelliJ
    - Eclipse
- Graphical interface
    - Rancher Dashboard
    - Headlamp projects
# 5 Pods
- A Pod is a collection of ==application containers== and ==volumes== running in the same execution environment. Pods, not containers, are the smallest deployable artifact in a Kubernetes cluster
    - In some cases, multiple containers is symbiotic and should exist together, like the initContainer
- ==How to decide if container should be together in 1 pod?==
    - “Will these containers work correctly if they land on different machines?” If the answer is no, a Pod is the correct grouping for the containers
- pods are defined in a ==manifest==
    - API server process these manifests and persist into the storage (etcd

- Create Pods
    - Imperative
  ```
      kubectl run kuard --generator=run-pod/v1  \ 
      --image=gcr.io/kuar-demo/kuard-amd64:blue
  ```

    - Declarative (manfiest):
        - Ex kuard-pod.yaml
            - can also include health checks, resource limits (is described as per container)
      ``` yaml
          apiVersion: v1
          kind: Pod
          metadata:
            name: kuard
          spec:
            containers:
              - image: gcr.io/kuar-demo/kuard-amd64:blue
                name: kuard
                ports:
                  - containerPort: 8080
                    name: http
                    protocol: TCP
      ```
        - kubectl apply -f kuard-pod.yaml

- Other commands
```terminal
ktl get pods 
ktl get pods -o wide | json | yaml

ktl describe pods [POD-NAME, ie kuard]

ktl delete pods/[POD-NAME, ie kuard]
kubectl delete -f kuard-pod.yaml

# Log
kubectl logs kuard
kubectl logs kuard --previous # previous instance of pod, to investigate crash

# Run commands in container with exec
ktl exec -it kuard --date
kubectl exec -it kuard -- ash. # it is for interactive, so you can log in

# Port forward
kubectl port-forward kuard 8080:8080

```


- Using Volumes With Pods
    - specs.volume : array defines all of the volumes that may be accessed by containers in the Pod manifest
    - volumeMounts: This array defines the volumes that are mounted into a particular container and the path where each volume should be mounted
    - IMOW: volume outlives the container life cycle, and can be consumed by other containers in the mod
  ```yaml
  apiVersion: v1
  kind: Pod
  metadata:
    name: kuard
  spec:
    volumes:
      - name: "kuard-data"
        hostPath:
          path: "/var/lib/kuard"
    containers:
      - image: gcr.io/kuar-demo/kuard-amd64:blue
        name: kuard
        volumeMounts:
          - mountPath: "/data"
            name: "kuard-data"
        ports:
          - containerPort: 8080
            name: http
            protocol: TCP
  ```

    - Patterns of using volumes
        - communication/synchronization between containers in a pod.  Pod uses an empyDir volume which is scoped to pod's life span.
        - Cache
        - Persistent volume: use protocols like NFS, iSCSI, Amazon Elastic Block Store, Google Persistent Disk
        - Mounting to host filesystem, like the /dev filesystem to perform raw block level access to device.  Use ==hostPath== volume
        - For more details, refer chapter 16
        - Example:
          ```yaml
          # Rest of pod definition above here
          volumes:
              - name: "kuard-data"
                nfs:
                  server: my.nfs.server.local
                  path: "/exports"
          ```

# 6 Labels and Annotations
### Labels
- Labels are key/value pairs that can be attached to Kubernetes objects such as Pods and ReplicaSets.
- Sample commands
```terminal
	kubectl run alpaca-prod \
	  --image=gcr.io/kuar-demo/kuard-amd64:blue \
	  --replicas=2 \
	  --labels="ver=1,app=alpaca,env=prod"

	kubectl get deployments --show-labels

	# Modify labels
	kubectl label deployments alpaca-test "canary=true"
	kubectl get deployments -L canary

	# Remove labels, with -
	kubectl label deployments alpaca-test "canary-"
```


### Label Selectors
- Selectors are used to filter k8s objects based on set of labels
- Commands
```yaml

	# First show all the labels
	kubectl get pods --show-labels
	# NAME                              ... LABELS
	# alpaca-prod-3408831585-4nzfb      ... app=alpaca,env=prod,ver=1,...

	kubectl get pods --selector="ver=2"
	kubectl get pods --selector="app=bandicoot,ver=2"
	kubectl get pods --selector="app in (alpaca,bandicoot)"

	kubectl get deployments --selector='!canary'
	kubectl get pods -l 'ver=2,!canary'
```




### Annotations
- Labels vs Annotations
    - annotations are metadata to assist tools and libraries

```yaml
		metadata:
		  annotations:
		    example.com/icon-url: "https://example.com/icon.png"
```

# 7 Service Discovery (Via K8s Service Object)
### 7.1 What is Service Discovery
- Service-discovery tools help solve the problem of finding which processes are listening at which addresses for which services
    - Domain Name System (DNS) is an example.  K8s system is much more dynamic, java programs often often have stale mapping.

### 7.2 The Service Object
- K8s service object deals with service discovery.
    - It is a way to crate a named label selector
- K8s deployment is an instance of a micro service

- Example of a K8s deployment
```terminal
$ **kubectl create deployment alpaca-prod \
  --image=gcr.io/kuar-demo/kuard-amd64:blue \
  --port=8080**
$ **kubectl scale deployment alpaca-prod --replicas 3**
$ **kubectl expose deployment alpaca-prod**
$ **kubectl create deployment bandicoot-prod \
  --image=gcr.io/kuar-demo/kuard-amd64:green \
  --port=8080**
$ **kubectl scale deployment bandicoot-prod --replicas 2**
  **kubectl expose deployment bandicoot-prod**
$ **kubectl get services -o wide**

NAME             CLUSTER-IP    ... PORT(S)  ... SELECTOR
alpaca-prod      10.115.245.13 ... 8080/TCP ... app=alpaca
bandicoot-prod   10.115.242.3  ... 8080/TCP ... app=bandicoot
kubernetes       10.115.240.1  ... 443/TCP  ... <none>

```

-  3 services
    -  k8s service is automatically created, to talk to the K8s API from each of the 2 apps
- expose command pulls the label selector and relevant ports from the deployment definition
- ==Cluster IP== is a virtual IP address the system will load balance across all of the Pods that are identified by the selector
- Port forward to one of the alpaca pods, from my mac with http://localhost:48858
```terminal
$ ALPACA_POD=$(kubectl get pods -l app=alpaca \
    -o jsonpath='{.items[0].metadata.name}')
$ kubectl port-forward $ALPACA_POD 48858:8080
```

##### Service DNS
- Cluster IP is virtual, so clients caching DNS is not an issue anymore.
    - Within a namespace, clients can just connect using the service name.  Like google.com
- Example of a full DNS:
    - alpaca-prod.default.svc.cluster.local
        - service name: alpaca-prod
        - namespace: default
        - svc: K8s can expost other types of DNS in the future
        - cluster.local: base domain for cluster; admins can modify

### 7.3 Looking Beyond the Cluster
- Previous section talks about how to communicate within cluster via services. How to talk to ==outside== the cluster? One way is with ==nodeport==
    - Enhances a service. In addition to cluster ip, a port is selected so other nodes in the cluster can forward traffic to that port in the service.
        - One can reach any service in any cluster node
    - We then use a load balancer to expose the service to outside the cluster.
- Example: change a service to nodeport
```terminal
kubectl edit service alpaca-prod

change spec.type from service to NodePort

kubectl describe service alpaca-prod
	Name:                   alpaca-prod
	Namespace:              default
	Labels:                 app=alpaca
	Annotations:            <none>
	Selector:               app=alpaca
	Type:                   NodePort
	IP:                     10.115.245.13
	Port:                   <unset> 8080/TCP
	NodePort:               <unset> 32711/TCP
	Endpoints:              10.112.1.66:8080,10.112.2.104:8080,10.112.2.105:8080
	Session Affinity:       None
	No events.
	
ssh <node> -L 8080:localhost:32711
```

### 7.4 Load Balancer Integrations
- Prereq: Update cluster to integrate with external loadbalancer.
- Edit k8s nodeport to be loadbalancer
    - kubectl edit service alpaca-prod
    - configure the cloud to create a new load balancer and direct it at nodes in your cluster
```terminal

kubectl describe service alpaca-prod
	Name:                   alpaca-prod
	Namespace:              default
	Labels:                 app=alpaca
	Selector:               app=alpaca
	Type:                   LoadBalancer
	IP:                     10.115.245.13
	LoadBalancer Ingress:   104.196.248.204
	Port:                   <unset>	8080/TCP
	NodePort:               <unset>	32711/TCP
	Endpoints:              10.112.1.66:8080,10.112.2.104:8080,10.112.2.105:8080
	Session Affinity:       None
	Events:
	  FirstSeen ... Reason                Message
	  --------- ... ------                -------
	  3m        ... Type                  NodePort -> LoadBalancer
	  3m        ... CreatingLoadBalancer  Creating load balancer
	  2m        ... CreatedLoadBalancer   Created load balancer

# service outslide of cluster can talk via the ip specified in LoadBalancer Ingress
```
### 7.5 Advanced Details
##### Endpoints
- Every k8s service is k8s endpoint
- Endpoints contains all the IP address for that service; it does not use clusterI ip address
- Applications can talk to K8s API to look up endpoints for a service.
    - Pods, nodes can change, but the endpoint always has the updated ips
```terminal
kubectl get endpoints alpaca-prod --watch
	NAME          ENDPOINTS                                            AGE
	alpaca-prod   10.112.1.54:8080,10.112.2.84:8080,10.112.2.85:8080   1m


# UPDATE service
$ kubectl delete deployment alpaca-prod
$ kubectl create deployment alpaca-prod \
  --image=gcr.io/kuar-demo/kuard-amd64:blue \
  --port=8080
$ kubectl scale deployment alpaca-prod --replicas=3

NAME          ENDPOINTS                                            AGE
alpaca-prod   10.112.1.54:8080,10.112.2.84:8080,10.112.2.85:8080   1m
alpaca-prod   10.112.1.54:8080,10.112.2.84:8080                    1m
alpaca-prod   <none>                                               1m
alpaca-prod   10.112.2.90:8080                                     1m
alpaca-prod   10.112.1.57:8080,10.112.2.90:8080                    1m
alpaca-prod   10.112.0.28:8080,10.112.1.57:8080,10.112.2.90:8080   1m

```


##### KubeProxy and Cluster IPs
- kubeproxy maintains a node's cluster ip (iptables).  It runs in every node.
    - kube-proxy watches for new services via the API server

### 7.6 Connecting With Other Environments


### 7.7 Cleanup


# 8 HTTP Load Balancing With ==k8s Ingress==
- For many users, service objects are sufficient.
    - Service objects operate at layer 4; it can only forward TCP and UDP connection.
        - service with cluster --> use nodeport, ie a unique port
        - service outside cluster --> use load balancer, but requires expensive loadbalancers
    - For http (layer 7) based services, we can do better
- K8s Ingress implements http based load balancing
    - achieves "Virtual hosting pattern": 1 IP address --> multiple services
    - If I have 5 underlying backends, clients all call my 1 IP. But based on the http connection, proxies the request to one of my 5 underlying backend services
### 8.1 Ingress Spec Vs Ingress Controller


### 8.2 Installing Contour


### 8.3 Using Ingress


### 8.4  Advance Ingress Topics and Gotchas



### 8.5 Alternate Ingress Implementation



### 8.6 Future of Ingress
- has some intersection with k8s service mesh, discuss later

# 9 ReplicaSets


# 10 Deployments

- K8s deployments
    - enables us to release a new version not tied to a new container image, ie artifacts, requirements.txt
        - K8s pods and replicasets are tied to a static container image
    - handles rollout process
    - support health checks
### 10.1 Your First Deployment
```yaml
	apiVersion: apps/v1
	kind: Deployment
	metadata:
	  name: kuard
	  labels:
	    run: kuard
	spec:
	  selector:
	    matchLabels:
	      run: kuard
	  replicas: 1
	  template:
	    metadata:
	      labels:
	        run: kuard
	    spec:
	      containers:
	      - name: kuard
	        image: gcr.io/kuar-demo/kuard-amd64:blue

$ kubectl create -f kuard-deployment.yaml

$ kubectl get deployments kuard \
  -o jsonpath --template {.spec.selector.matchLabels}
	{"run":"kuard"}

$ kubectl scale deployments kuard --replicas=2

$ **kubectl get replicasets --selector=run=kuard**
	NAME              DESIRED   CURRENT   READY     AGE
	kuard-1128242161  2         2         2         13m

```

### 10.2 Creating Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  annotations:
    deployment.kubernetes.io/revision: "1"
  creationTimestamp: null
  generation: 1
  labels:
    run: kuard
  name: kuard
spec:
  progressDeadlineSeconds: 600
  replicas: 1
  revisionHistoryLimit: 10
  selector:
    matchLabels:
      run: kuard
  strategy:
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
    type: RollingUpdate
  template:
    metadata:
      creationTimestamp: null
      labels:
        run: kuard
    spec:
      containers:
      - image: gcr.io/kuar-demo/kuard-amd64:blue
        imagePullPolicy: IfNotPresent
        name: kuard
        resources: {}
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
      dnsPolicy: ClusterFirst
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      terminationGracePeriodSeconds: 30
status: {}
```
- new sections
    - strategy
    -
### 10.3 Managing Deployments
- kubectl describe deployments kuard

### 10.4 Updating Deployments
- could also use helm; k8s seems to be at a lower level
```terminal
edit the deployment file

ktl apply -f deployment.yaml

ktl get deployments kuard

kubectl rollout status deployments kuard

kubectl rollout pause deployments kuard

kubectl rollout history deployment kuard
kubectl rollout history deployment kuard --revision=2

# undo a rollout
kubectl rollout undo deployments kuard


```

### 10.5 Deployment Strategies
- Two strategies: recreate and rollingUpdate
##### Recreate Strategy
- It simply updates the ReplicaSet it manages to use the new image and terminates all of the Pods associated with the Deployment.
- Simple, fast, and simple
    - but will result in workload downtime.
    - Should only be used for test deployments

##### RollingUpdate Strategy
- Enables one to roll out a new version of your service while it is still receiving user traffic, without any downtime

##### Slowing Rollouts to Ensure Health



### 10.6 Deleting a Deployment
```terminal
kubectl delete deployments kuard

kubectl delete -f kuard-deployment.yaml
```

### 10.7 Monitoring a Deployment
-  the status of the Deployment will transition to a failed state. This status can be obtained from the status.conditions array, where there will be a Condition whose Type is Progressing and whose Status is False

# 11 Daemon Sets
- Purpose is to schedule a single Pod on every node within the cluster. land some sort of agent or daemon on each node
### 11.1 DaemonSet Scheduler
- By default, a DaemonSet will create a copy of a Pod on every node unless a node selector is used

### 11.2 Creating DaemonSets
- Ex: Create a fluentd logging agent on every node
```yaml
	spec:
	  selector:
	    matchLabels:
	      app: fluentd
	  template:
	    metadata:
	      labels:
	        app: fluentd
	    spec:
	      containers:
	      - name: fluentd
	        image: fluent/fluentd:v0.14.10
	        resources:
	          limits:
	            memory: 200Mi
	          requests:
	            cpu: 100m
	            memory: 200Mi
	        volumeMounts:
	        - name: varlog
	          mountPath: /var/log
	        - name: varlibdockercontainers
	          mountPath: /var/lib/docker/containers
	          readOnly: true
	      terminationGracePeriodSeconds: 30
	      volumes:
	      - name: varlog
	        hostPath:
	          path: /var/log
	      - name: varlibdockercontainers
	        hostPath:
	          path: /var/lib/docker/containers

kubectl apply -f fluentd.yaml
kubectl describe daemonset fluentd

kubectl get pods -l app=fluentd -o wide
NAME            READY   STATUS    RESTARTS   AGE   IP             NODE
fluentd-1q6c6   1/1     Running   0          13m   10.240.0.101   k0-default...
fluentd-mwi7h   1/1     Running   0          13m   10.240.0.80    k0-default...
fluentd-zr6l7   1/1     Running   0          13m   10.240.0.44    k0-default...
```


### 11.3 Limiting DaemSets to Specific Nodes
- Add labels to nodes
    - **kubectl label nodes k0-default-pool-35609c18-z7tb ssd=true**
    - **kubectl get nodes --selector ssd=true**
- use the metadata.labels fiels
```yaml
	apiVersion: apps/v1
	kind: "DaemonSet"
	metadata:
	  labels:
	    app: nginx
	    ssd: "true"
	  name: nginx-fast-storage
	spec:
	  selector:
	    matchLabels:
	      app: nginx
	      ssd: "true"
	  template:
	    metadata:
	      labels:
	        app: nginx
	        ssd: "true"
	    spec:
	      nodeSelector:
	        ssd: "true"
	      containers:
	        - name: nginx
	          image: nginx:1.10.0
```

### 11.4 Upading a DaemsoneSet


### 11.5 Deleting a DaemonSet



# 12 Jobs
- Previously, we focus on long-running processes.
- K8s job pods runs until successful termination, ie exit with 0
    - In contrast, non-job pods will keep restarting with exit with 0
### Job Patterns
1. One shot
    1. use case: database migrations
    2. A single Pod running once until successful termination
2. Parallel fixed completions
    1. use case: Multiple Pods processing a set of work in
    2. One or more Pods running one or more times until reaching a fixed completion count
3. Worker queue: parallel jobs
    1. use case: Multiple Pods processing from a centralized work queue
    2. One or more Pods running once until successful termination


### Cronjobs
- Schedule a job to run at a certain interval

- Example:
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: example-cron
spec:
  # Run every fifth hour
  schedule: "0 */5 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: batch-job
            image: my-batch-image
          restartPolicy: OnFailure
```


# 13 ConfigMaps and Secrets
### 13.1 ConfigMaps
- A set of variables that can be used when defining the environment or command line for your containers
    - If a container image reads these environment variables, we can reuse the image for multiple workloads

##### Creating a ConfigMap
- Example
```
# my-config.txt
# This is a sample config file that I might use to configure an application
parameter1 = value1
parameter2 = value2

$ kubectl create configmap my-config \
  --from-file=my-config.txt \
  --from-literal=extra-param=extra-value \
  --from-literal=another-param=another-value

$ kubectl get configmaps my-config -o yaml
apiVersion: v1
data:
  another-param: another-value
  extra-param: extra-value
  my-config.txt: |
    # This is a sample config file that I might use to configure an application
    parameter1 = value1
    parameter2 = value2
kind: ConfigMap
metadata:
  creationTimestamp: ...
  name: my-config
  namespace: default
  resourceVersion: "13556"
  selfLink: /api/v1/namespaces/default/configmaps/my-config
  uid: 3641c553-f7de-11e6-98c9-06135271a273
```

##### Using a ConfigMap
- 3 main ways to use configMaps
    1. File System:
        1. You can mount a ConfigMap into a Pod. A file is created for each entry based on the key name. The contents of that file are set to the value.
    2. Environment Variable
        1. A ConfigMap can be used to dynamically set the value of an environment variable.
    3. Command line argument
        1. Kubernetes supports dynamically creating the command line for a container based on ConfigMap values.

- Example of all 3
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kuard-config
spec:
  containers:
    - name: test-container
      image: gcr.io/kuar-demo/kuard-amd64:blue
      imagePullPolicy: Always
      command:
        - "/kuard"
        - "$(EXTRA_PARAM)"
      env:
        # An example of an environment variable used inside the container
        - name: ANOTHER_PARAM
          valueFrom:
            configMapKeyRef:
              name: my-config
              key: another-param
        # An example of an environment variable passed to the command to start
        # the container (above).
        - name: EXTRA_PARAM
          valueFrom:
            configMapKeyRef:
              name: my-config
              key: extra-param
      volumeMounts:
        # Mounting the ConfigMap as a set of files
        - name: config-volume
          mountPath: /config
  volumes:
    - name: config-volume
      configMap:
        name: my-config
  restartPolicy: Never
```

- Filesystem:
  - We have to specify where this gets mounted into the kuard container with a ==volumeMount==  In this case, we are mounting it at /config
- Environment
    - Environment variables are specified with a special ==valueFrom== member.


### 13.2 Secrets
- For configuration that are sensitive.
##### Creating Secrets
- Secrets hold 1 or more elements as a collection of key/value pairs
- Steps
    1. Download the secrets from somewhere
        1. $ **curl -o kuard.crt  https://storage.googleapis.com/kuar-demo/kuard.crt**
        2. **curl -o kuard.key https://storage.googleapis.com/kuar-demo/kuard.key**
    2. create the k8s secret
       ```
        kubectl create secret generic kuard-tls \
         --from-file=kuard.crt \
         --from-file=kuard.key**
  ``
    3. Describe the secret
  ```terminal
  kubectl describe secrets kuard-tls**
  
  ```


##### Consuming Secrets
- The kuard-tls Secret contains two data elements: kuard.crt and kuard.key. Mounting the kuard-tls Secrets volume to /tls results in the following files:

- Ex:
    - first, create the volume tls-certs, which is associated with the kuard-tls secret created above
    - then, container uses volumeMounts, which points to the volume tls-certs
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kuard-tls
spec:
  containers:
    - name: kuard-tls
      image: gcr.io/kuar-demo/kuard-amd64:blue
      imagePullPolicy: Always
      volumeMounts:
      - name: tls-certs
        mountPath: "/tls"
        readOnly: true
  volumes:
    - name: tls-certs
      secret:
        secretName: kuard-tls
```

### 13.3 Naming Constraints


### 13.4 Managing ConfigMaps and Secrets




# 14 Role Based Access For K8s
- Role-based access control provides a mechanism for restricting both access to and actions on Kubernetes APIs
    - if you are focused on hostile multitenant security, RBAC by itself is sufficient to protect you
    - However, anyone who can run arbitrary code inside K8s cluster an obtain root privilege to entire cluster
- RBAC concept
    - Authentication: Authentication provides the identity of the caller issuing the request.
        - Using authentication provide, like Azure active directory
        - K8s has no built in identity store
    - Authorization: what is the user authorized to perform?

### 14.1 Role Based Access Control
##### Identity in K8s
- Every request to Kubernetes is associated with some identity
- Kubernetes uses a generic interface for authentication providers
    - HTTP Basic Authentication (largely deprecated)
    - x509 client certificates
    - Static token files on the host
    - Cloud authentication providers, such as Azure Active Directory and AWS Identity and Access Management (IAM)
    - Authentication webhooks

##### Understanding Roles and Role Bindings
- Once Kubernetes knows the identity of the request, it needs to determine if the request is authorized for that user. To achieve this, it uses roles and role bindings.
- A role is a set of abstract capabilities.
    - For example, the appdev role might represent the ability to create Pods and Services.
- A role binding is an assignment of a role to one or more identities.
    - Thus, binding the appdev role to the user identity alice indicates that Alice has the ability to create Pods and Services.

##### Roles and Role Bindings in K8s
- Role and role bindings:
    - name space level
        - role, role binding
    - cluster level
        - clusterRole, clusterRoleBinding
- Example: Role
```yaml
	kind: Role
	apiVersion: rbac.authorization.k8s.io/v1
	metadata:
	  namespace: default
	  name: pod-and-services
	rules:
	- apiGroups: [""]
	  resources: ["pods", "services"]
	  verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
```

- Example: Role Binding
```yaml
	apiVersion: rbac.authorization.k8s.io/v1
	kind: RoleBinding
	metadata:
	  namespace: default
	  name: pods-and-services
	subjects:
	- apiGroup: rbac.authorization.k8s.io
	  kind: User
	  name: alice
	- apiGroup: rbac.authorization.k8s.io
	  kind: Group
	  name: mydevs
	roleRef:
	  apiGroup: rbac.authorization.k8s.io
	  kind: Role
	  name: pod-and-services
```


### 14.2 Techniques for Managing RBAC
##### Testing Authorization with can-i
```terminal
kubectl auth can-i create pods

# test subresources like logs or port-forwarding with the --subresource
kubectl auth can-i get pods --subresource=logs
```


##### Managing RBAC in Source Control
- The kubectl command-line tool provides a reconcile command that operates somewhat like kubectl apply and will reconcile a set of roles and role bindings with the current state of the cluster. You can run:
```terminal
$ kubectl auth reconcile -f some-rbac-config.yaml

```


### 14.3 Advanced Topics
##### Aggregating ClusterRoles

##### Using Groups for Bindings


# 15 Service Mesh
- Previous K8s network primitives: services, ingress
- Both Service and Ingress resources have label selectors that route traffic to a particular set of Pods, but beyond that there is comparatively little in the way of additional capabilities that these resources bring. As an HTTP load balancer, Ingress goes a little beyond this, but the challenge of defining a common API that fits a wide variety of different existing implementations limits the capabilities in the Ingress API
- Service mesh APIs provide additional ==cloud native== networking capabilities. See each section below to see what cloud native means.

##### Feature: Encryption and Authentication with Mutual TLS
- Service mesh on your Kubernetes cluster automatically provides encryption to network traffic between every Pod in the cluster
    - consistent implementation across all deployments
##### Feature: Traffic Shaping
- Traffic shaping = Routing of requests to different service implementations based on the characteristics of the request
    - Isn't this just "virtual hosting"?
- Example: A/B Testing
    - Instead of writing code to implement your experiment, or deploying an entirely new copy of your application on new infrastructure, you declaratively define the parameters of the experiment (10% of traffic to version Y, 90% of traffic to version X), and the service mesh implements it for you.
    - Was this the feature flag we saw in NPD?
##### Feature: Introspection
- Instead of seeing a flurry of requests to a bunch of different microservices, the developer can see a single aggregate request that defines the user experience of their complete application.
    - Is this the aggregate logging solution I wanted ?
- The service mesh is implemented once for an entire cluster. This means that the same request tracing works no matter which team developed the service. The monitoring data is entirely consistent across all of the different services joined together by a cluster-wide service mesh.
-
##### Do You Really Need Service Mesh?
- A service mesh is a distributed system that adds complexity to your application design. The service mesh is deeply integrated into the core communication of your microservices. When a service mesh fails, your entire application stops working.

##### Service Mesh Landscape
- Open source:
    - Istio,  Linkerd, Consul Connect, Open Service Mesh, and others.
- There are also proprietary meshes like AWS App Mesh.

# 16 Integrating Storage Solutions And K8s


# 17 Extending K8s


# 18 Accessing K8s From Common Programming Language


# 19 Securing Applications in K8s


# 20 Policy and Governance For K8s Clusters


# 21 Multi cluster Application Deployments



# 22 Organizing Your Application

### 22.1 Principles to Guide Us
##### Filesystems as the Source of Truth
- Rather than viewing the state of the cluster—the data in `etcd`—as the source of truth, it is optimal to view the filesystem of YAML objects as the source of truth for your application.

##### Role of Code Review
- Ensuring that at least two people look at any configuration change significantly decreases the probability of errors.

##### Feature Gates
- Should application source code and k8s configuration files (aka source of truth) be in separate git repositories?
    -  Yes, the perspectives of the builder versus those of the deployer are different enough that this separation of concerns makes sense

- How do you bridge the development of new features in source control with the deployment of those features into a production environment?
    - with feature flags
  ``` python
  if (featureFlags.myFlag) {
      // Feature implementation goes here
  }
  ```


### 22.2 Managing Your Application in Source Control
##### File System Layout
- organize your application is the semantic component or layer (for instance, frontend or batch work queue)
    - Rationale: Goal is to be able scale teams, where each team is responsible for a component and layer

```
frontend/
   frontend-deployment.yaml
   frontend-service.yaml
   frontend-ingress.yaml
service-1/
   service-1-deployment.yaml
   service-1-service.yaml
   service-1-configmap.yaml
   
```

##### Managing Periodic Versions of the Deployment
- It is handy to be able to simultaneously store and maintain multiple revisions of your configuration.
- Options
1. Use (git) tags, branches, and source-control features. This is convenient because it maps to the way people manage revisions in source control, and leads to a more simplified directory structure
2. Clone the configuration within the filesystem and use directories for different revisions.


- Option 1 in more details
    - When you are ready for a release, you place a source-control tag (such as git tag v1.0) in the configuration source-control system. The tag represents the configuration used for that version, and the HEAD of source control continues to iterate forward.
    - Updating the release configuration is somewhat more complicated, but the approach models what you would do in source control. First, you commit the change to the HEAD of the repository. Then you create a new branch named v1 at the v1.0 tag. You cherry-pick the desired change onto the release branch (git cherry-pick edit), and finally, you tag this branch with the v1.1 tag to indicate a new point release.


- Option 2 in more details
```
	frontend/
	  v1/
		frontend-deployment.yaml
		frontend-service.yaml
	  current/
		frontend-deployment.yaml
		frontend-service.yaml
	service-1/
	  v1/
		 service-1-deployment.yaml
		 service-1-service.yaml
	  v2/
		 service-1-deployment.yaml
		 service-1-service.yaml
	  current/
		 service-1-deployment.yaml
		 service-1-service.yaml
```

### 22.3 Structuring Application for Development, Testing, and Deployment


### 22.4 Parameterizing Application with Templates
- Parameterized environments use templates for the bulk of their configuration, but they mix in a limited set of parameters to produce the final configuration.

##### Parameterizing with Helm and Templates
- helm uses "mustache syntax"
    - Ex: deployment.yaml
  ```yaml
  metadata:
    name: {{ .Release.Name }}-deployment
  ```

    - Values.yaml
        - we have separate values.yaml for each of our environments, ie dev, stage, test
  ```yaml
  Release:
    Name: my-release
  ```

- helm uses values.yaml to populate the k8s resource file.
  ``

### 22.5 Deploying Application Around the World
- Adding multiple regions to your configuration is identical to adding new life cycle stages
```terminal
frontend/
  staging/
    templates -> ../v3/
    parameters.yaml
  eastus/
    templates -> ../v1/
    parameters.yaml
  westus/
    templates -> ../v2/
    parameters.yaml
  ...
```

# Appendix: Building Your Own K8s Cluster