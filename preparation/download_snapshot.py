from huggingface_hub import snapshot_download

# this will create exactly this folder, with config.json inside
local_dir = snapshot_download(
    repo_id="zelaki/eq-vae",
    local_dir="./zelaki-eq-vae",
    resume_download=True
)