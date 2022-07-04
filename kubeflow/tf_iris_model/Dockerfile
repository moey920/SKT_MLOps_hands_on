FROM tensorflow/tensorflow:2.8.0
ENV PYTHONUNBUFFERED 1
LABEL org.opencontainers.image.source=https://github.com/moey920/SKT_MLOps_hands_on
RUN mkdir /code
WORKDIR /code
COPY ./ /code/
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt