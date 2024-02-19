ARG VARIANT="3.11"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# [Option] Install Node.js
ARG INSTALL_NODE="true"
ARG NODE_VERSION="lts/*"
RUN if [ "${INSTALL_NODE}" = "true" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# install ffmpeg to generate mpeg4 movies, nodejs, and pygame to run jupyterlab extensions
RUN apt update && apt install -y ffmpeg nodejs && rm -rf /var/lib/{apt,dpkg,cache,log}

RUN export DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir --upgrade pip

# add repo directory to PYTHONPATH
# ENV PYTHONPATH="${PYTHONPATH}:/workspace/repo"
ENV PYTHONPATH="${PYTHONPATH}:$(pwd)"

# create folder for jupyter notebook extensions and set the appropiate permissions
RUN mkdir /usr/local/etc/jupyter && chown -R vscode:vscode /usr/local/etc/jupyter
RUN mkdir /usr/local/share/jupyter && chown -R vscode:vscode /usr/local/share/jupyter
