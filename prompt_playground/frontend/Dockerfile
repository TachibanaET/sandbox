FROM node:18-bullseye-slim
LABEL maintainer="TachibanaET <https://github.com/TachibanaET>"

# Install dependencies
ARG PROXY
ENV HTTP_PROXY ${PROXY}
ENV HTTPS_PROXY ${PROXY}
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ Asia/Tokyo

RUN apt update && apt -y upgrade
RUN apt -y install vim \
    wget \
    curl \
    git \
    gcc \
    cmake \
    unzip \
    tmux \
    sudo \
    tzdata


# Create user & add to sudoers
ARG UID
ARG GID
ARG UNAME

RUN groupadd -g ${GID} ${UNAME} && \
    useradd -u ${UID} -g ${UNAME} -G sudo  -m ${UNAME} && \
    echo "${UNAME}:${UNAME}" | chpasswd && \
    echo "${UNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV UID ${UID}
ENV GID ${GID}
ENV UNAME ${UNAME}


# Change Permission
COPY ./set_volume_permission.sh /home/${UNAME}/
RUN chmod +x /home/${UNAME}/set_volume_permission.sh

RUN /home/${UNAME}/set_volume_permission.sh

# Install React
WORKDIR /home/${UNAME}/workspace/source
COPY ./frontend/source/package.json /home/${UNAME}/workspace/source/
RUN npm install -g create-react-app
RUN npm install
CMD [ "npm", "start" ]