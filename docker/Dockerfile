FROM csiroeasi/datacube-drivers-test:3d.latest

# Based on https://github.com/opendatacube/datacube-core/blob/develop/docker/Dockerfile

# Prep postgres database
RUN  install --owner postgres --group postgres -D -d /var/run/postgresql /srv/postgresql \
  && sudo -u postgres "$(find /usr/lib/postgresql/ -type f -name initdb)" -D "/srv/postgresql" --auth-host=md5 --encoding=UTF8

RUN groupadd --gid 1000 odc \
  && useradd --gid 1000 \
  --uid 1000 \
  --create-home \
  --shell /bin/bash -N odc \
  && adduser odc users \
  && adduser odc sudo \
  && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
  && true

# Ensure pip is up to date and install additional dependencies
RUN pip install --upgrade pip \
    && rm -rf $HOME/.cache/pip

# Copy and Install driver
WORKDIR /opt/datacube_zarr
COPY . .
RUN pip install --upgrade -e .[test]

# Initialisation scripts for DB creation
COPY docker/with_bootstrap.sh /usr/local/bin/with_bootstrap.sh

ENTRYPOINT ["/bin/tini", "-s", "--", "/usr/local/bin/with_bootstrap.sh"]

CMD ["bash"]
