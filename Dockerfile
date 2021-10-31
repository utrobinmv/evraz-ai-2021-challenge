FROM nvidia/cuda:11.0-base

ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Install system libraries required by OpenCV.
RUN apt-get update \
 && apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6

RUN apt-get install -y bash gcc

RUN apt-get install -y python3-pip

COPY requirements.txt ${APPDIR}/

RUN pip install --no-cache-dir -r ${APPDIR}/requirements.txt

ARG APPDIR=/app

WORKDIR ${APPDIR}

COPY  *.py ${APPDIR}/

COPY yolov4_pytorch ${APPDIR}/yolov4_pytorch

COPY configs ${APPDIR}/configs

COPY data ${APPDIR}/data

COPY dataset ${APPDIR}/dataset

COPY dataset/train/annotations/COCO_json ${APPDIR}/dataset/train/annotations/COCO_json

COPY weights ${APPDIR}/weights

CMD ["python3", "02-recognition-test.py"]
