FROM nvcr.io/nvidia/pytorch:23.08-py3

WORKDIR /workspace

RUN rm -rf *
COPY . squad-2.0
RUN pip3 install -r squad-2.0/container-requirements.txt
EXPOSE 8888
