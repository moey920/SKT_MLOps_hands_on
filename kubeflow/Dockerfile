# FROM tensorflow/tensorflow:2.8.0-gpu
FROM tensorflow/tensorflow:2.8.0
ENV PYTHONUNBUFFERED 1
LABEL org.opencontainers.image.source=https://github.com/Musma/automl
RUN mkdir /code
WORKDIR /code
# CMD ["python", "/code/Common/influx.py",]
COPY ./ /code/
RUN chmod -R 777 /code
RUN chmod -R 777 /mnt
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt