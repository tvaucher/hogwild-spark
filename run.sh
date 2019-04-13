GROUP_NAME=cs449g13
APP_NAME=spark-svm
NB_EXECUTOR=4
IMAGE=tvaucher/svm-spark:run1

while getopts ":n:" opt; do
  case $opt in
    n) NB_EXECUTOR="$OPTARG";; # number workers
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

kubectl delete pod $APP_NAME
spark-submit \
    --master k8s://https://10.90.36.16:6443 \
    --deploy-mode cluster \
    --name $APP_NAME \
    --conf spark.executor.instances=$NB_EXECUTOR \
    --conf spark.kubernetes.namespace=$GROUP_NAME \
    --conf spark.kubernetes.driver.pod.name=$APP_NAME \
	--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.volume1.options.claimName=$GROUP_NAME-scratch \
	--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.volume1.options.claimName=$GROUP_NAME-scratch \
    --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.volume1.mount.path=/data \
	--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.volume1.mount.path=/data \
    --conf spark.kubernetes.container.image=$IMAGE \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    local:///opt/spark/work-dir/hogwild.py