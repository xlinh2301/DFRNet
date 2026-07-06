# Workflow

- Remote dev target: `ssh linhnx_gmoe`, working dir `/media/gmo/data/linhnx/workspace/DRNet` (git repo, remote `gmoe` at `/media/gmo/data/linhnx/workspace/DRNet.git`).
- All GPU work (train/infer) runs inside the `MLR_LinhNX` docker container on that host (`docker exec -it MLR_LinhNX bash`), which mounts the repo at `/workspace/DRNet`.
- To ship local changes: commit locally, `git push gmoe main`, then on the server `cd /media/gmo/data/linhnx/workspace/DRNet && git pull`.
- Spec-driven changes are tracked with OpenSpec (`openspec/` dir, `/opsx:*` commands).
