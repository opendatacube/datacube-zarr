#!/bin/bash

# Based on https://github.com/opendatacube/datacube-core/blob/develop/docker/assets/with_bootstrap
launch_db () {
    local pgdata="${1:-/srv/postgresql}"
    local dbuser="${2:-odc}"
    local bin=/usr/lib/postgresql/12/bin

    [ -e "${pgdata}/PG_VERSION" ] || {
        sudo -u postgres "${bin}/initdb" -D "${pgdata}" --auth-host=md5 --encoding=UTF8
    }

    sudo -u postgres "${bin}/pg_ctl" -D "${pgdata}" -l "${pgdata}/pg.log" start

    sudo -u postgres createuser --superuser "${dbuser}"
    sudo -u postgres createdb "${dbuser}"
    sudo -u postgres createdb datacube
    sudo -u postgres createdb agdcintegration
}

# Become `odc` user with UID/GID compatible to datacube-core volume
#  If Running As root
#    launch db server
#    If outside volume not owned by root
#       change `odc` to have compatible UID/GID
#       re-exec this script as odc user

[[ $UID -ne 0 ]] || {
    [[ "${SKIP_DB:-no}" == "yes" ]] || {
        launch_db /srv/postgresql odc > /dev/null 2> /dev/null || {
            echo "WARNING: Failed to launch db, integration tests might not run"
        }
    }

    target_uid=$(stat -c '%u' .)
    target_gid=$(stat -c '%g' .)

    [[ $target_uid -eq 0 ]] || {
        groupmod --gid "${target_gid}" odc
        usermod --gid "${target_gid}" --uid "${target_uid}" odc
        chown -R odc:odc /home/odc/
        exec sudo -u odc -E -H bash "$0" "$@"
    }
}

[[ $UID -ne 0 ]] || echo "WARNING: Running as root"

cat <<EOL > $HOME/.datacube_integration.conf
[datacube]
db_hostname:
db_username: odc
db_database: agdcintegration
index_driver: default

[no_such_driver_env]
db_hostname:
index_driver: no_such_driver
EOL

[ -z "${1:-}" ] || {
    exec "$@"
}
