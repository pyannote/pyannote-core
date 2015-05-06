FROM stackbrew/ubuntu:12.04
MAINTAINER Hervé Bredin <bredin@limsi.fr>

RUN apt-get update
RUN apt-get install -y python-pip python-dev build-essential python-numpy
RUN apt-get install -y gfortran libblas-dev liblapack-dev

ADD . /src
RUN pip install numexpr
RUN pip install /src

CMD ['python']