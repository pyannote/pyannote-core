FROM pyannote/base
MAINTAINER Hervé Bredin <bredin@limsi.fr>

RUN pip install pyannote.core
RUN pip install pyannote.core[notebook]
