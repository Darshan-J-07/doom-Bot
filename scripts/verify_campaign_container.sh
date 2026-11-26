#!/usr/bin/env bash
set -euo pipefail

checkout=$(realpath -- "$1")
results=$(realpath -m -- "$2")
docker='/mnt/c/Users/Darshan/AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe'
image=devforge-doom:phase2-official-recipe

# Docker Desktop's per-distro mount service may be disabled. Stage only the
# clean candidate checkout on the Windows filesystem for a read-only bind mount.
staging=$(mktemp -d -p /mnt/c/Users/Darshan/AppData/Local/Temp doom-nexus-candidate.XXXXXX)
cp -a --no-preserve=ownership,mode "$checkout/." "$staging/"
win_path=$(wslpath -w "$staging")
mkdir -p -- "$results"
sha256sum "$checkout/submission/weights/best_model.pt" "$staging/submission/weights/best_model.pt" > "$results/container-weights.sha256"
"$docker" image inspect "$image" --format '{{.Id}}' > "$results/container-image.txt"

failed=0
for device in cpu cuda; do
    gpu=()
    if [[ "$device" == cuda ]]; then gpu=(--gpus all); fi
    log="$results/container-$device.log"
    exit_code=0
    "$docker" run --rm --pull never --network none --read-only --user 1000:1000 \
        --cpus 1 --memory 10240m --memory-swap 10240m --pids-limit 512 \
        --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:exec,size=2g --shm-size 2g \
        --workdir /tmp --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
        --env GIT_CONFIG_COUNT=1 --env GIT_CONFIG_KEY_0=safe.directory --env GIT_CONFIG_VALUE_0=/candidate \
        --mount "type=bind,source=$win_path,target=/candidate,readonly" "${gpu[@]}" \
        --entrypoint /opt/venv/bin/python "$image" /candidate/validate.py --repo /candidate \
        --in-docker --map /candidate/maps/two_rooms.wad --device "$device" > "$log" 2>&1 || exit_code=$?
    printf '%s\n' "$exit_code" > "$results/container-$device.exit"
    cat "$log"
    if ((exit_code)); then failed=1; fi
done
exit "$failed"
