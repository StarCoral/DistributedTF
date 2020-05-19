FROM tensorflow/tensorflow:latest-py3 
MAINTAINER Riya

RUN pip3 install tensorflow==1.5.0

RUN pip3 uninstall numpy -y && \
    pip3 install numpy==1.16

ADD ./tensor2.py ./tmp/
