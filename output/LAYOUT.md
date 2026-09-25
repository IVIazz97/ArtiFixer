# output/ layout

Grouped per method, then per scene (regrouped 2026-09-23 by `jobs/regroup_by_method.py`, which
also rewrote symlink targets and the absolute paths inside split.json/parsed.yaml).

```
output/
  recon/<scene>/        3dgrut_input, 3dgrut_runs, recon_results, captions, metric_alignment,
                        split.json, selected_{images.txt,indices.json}   ($RECON in jobs/common.sh)
  flashsplat/<scene>/   FlashSplat labels/contribution/hit_count + foreground/background renders
  flashsplat/garden_vase/   garden only: the vase on the table merged into the object (extrude_labels.py)
  artifixer/<scene>/    v0..v9, propagation/, af3d/     <- our own removal + refinement
  aurafusion/<scene>/   base/ + one dir per variant (AuraFusion360 port)
  inpaint360gs/<scene>/ base/ + one dir per variant (Inpaint360GS port)
  jobs/ logs/ panels/ smoke/
  mip360/               compatibility symlinks to the old paths; delete once nothing needs them
```

`jobs/common.sh` exports `OUT=$AF/output` and `RECON=$OUT/recon`, so a job addresses a method's
outputs as `$OUT/<method>/$SCENE` and the reconstruction as `$RECON/$SCENE`.

## ArtiFixer removal variants

| path | what |
|---|---|
| `artifixer/<scene>/vN/input/` | removal split fed to ArtiFixer (renders, opacity, hole masks, reference views, split.json) |
| `artifixer/<scene>/vN/artifixer/` | single-rollout ArtiFixer 1.3B output (`.../frames/batch_0000/pred`) |
| `artifixer/<scene>/propagation/<seed>/` | progressive propagation (anchors = photo outside mask + ArtiFixer inside) |
| `artifixer/<scene>/propagation/lama_seed_frames/` | LaMa-inpainted seed frames |
| `artifixer/<scene>/af3d/<seed>/` | ArtiFixer3D + ArtiFixer3D+ runs on a propagation result |

Variants (hole = SAM2 mask ∪ projected object opacity, dilated):

| v | segmentation | hole | opacity in hole | reference views | notes |
|---|---|---|---|---|---|
| v0 | flashsplat | mask +12px | 0 | photo patched with render | fills with a new object |
| v1 | flashsplat | mask +12px | render | photo patched with render | keeps smudge |
| v2 | flashsplat | mask +12px | render | background render | keeps smudge |
| v3 | flashsplat | mask +12px | 0 | background render | kitchen clean; garden fake table |
| v4 | flashsplat | hull +30px | 0 | background render | garden: block under vase |
| v5 | flashsplat_vase | mask +12px | 0 | background render | garden: clean early, ring after ~frame 28 |
| v6 | as v5/v3 | as base | full (LaMa-filled render) | LaMa-filled render | |
| v7 | as v5/v3 | as base | 0 (LaMa-filled render) | LaMa-filled render | |
| v8 | vase / flashsplat | hull +40px | 0 | background render | |
| v9 | vase / flashsplat | hull +80px | 0 | background render | |

Base for v6/v7: garden v5, kitchen v3. Job scripts: `jobs/stage*.sbatch`. Comparison panels:
`panels/`. The baseline ports have their own job scripts, `jobs/aurafusion/run.sbatch` and
`jobs/inpaint360gs/run.sbatch`, and their own READMEs in the repo root.
