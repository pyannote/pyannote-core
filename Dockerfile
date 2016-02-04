FROM pyannote/base
MAINTAINER Hervé Bredin <bredin@limsi.fr>

RUN DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    graphviz \
    libgraphviz-dev

RUN pip install pyannote.core

RUN pip install pyannote.core[notebook]
