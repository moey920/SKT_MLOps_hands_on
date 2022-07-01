#!/bin/bash
if [ $# -ne 1 ]
then
    echo "usage: $0 <docker hub ID>"
    exit 1
fi

USER=$1

IMG=data


echo "===================================================================================================="
echo "remove the existing images"
echo "----------------------------s----------------------------------------------------------${IMG}"
docker image rm ${IMG}
docker image rm ${USER}/${IMG}


echo "===================================================================================================="
echo "build ${IMG}"
echo "----------------------------------------------------------------------------------------------------"
pwd
docker build --no-cache -t ${IMG} .
docker tag ${IMG}:latest ${USER}/${IMG}:latest
docker push ${USER}/${IMG}:latest