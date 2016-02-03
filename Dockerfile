FROM pyannote/base
MAINTAINER Hervé Bredin <bredin@limsi.fr>

ADD . /src
RUN pip install /src
