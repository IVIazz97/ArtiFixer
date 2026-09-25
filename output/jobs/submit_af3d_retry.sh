#!/bin/bash
# Retry submitting the ArtiFixer3D jobs (chained after their propagation jobs) until Slurm accepts them.
cd /leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/jobs
declare -A DEP=([F_garden_lama]=58323776 [F_kitchen_lama]=58323781 [F_garden_v5]=58323058 [F_kitchen_v3]=58323094)
declare -A EXP=([F_garden_lama]=SCENE=garden,PROP=lama [F_kitchen_lama]=SCENE=kitchen,PROP=lama [F_garden_v5]=SCENE=garden,PROP=v5 [F_kitchen_v3]=SCENE=kitchen,PROP=v3)
for i in $(seq 1 720); do
  for name in "${!DEP[@]}"; do
    if out=$(sbatch --dependency=afterok:${DEP[$name]} --job-name=$name --export=ALL,${EXP[$name]} stageF_af3d.sbatch 2>&1); then
      echo "$(date) $name: $out"; unset "DEP[$name]"
    fi
  done
  [ ${#DEP[@]} -eq 0 ] && { echo "$(date) all submitted"; exit 0; }
  sleep 120
done
echo "$(date) gave up; remaining: ${!DEP[*]}"
