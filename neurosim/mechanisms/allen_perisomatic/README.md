# Allen "Biophysical - perisomatic" channel models

These NMODL files are the ion-channel models used by the Allen Institute's
*Biophysical – perisomatic* neuron models (Allen Cell Types Database,
neuronal model template `329230710`). They were downloaded from the Allen
Brain Map API (`https://api.brain-map.org/api/v2/well_known_file_download/<id>`)
on 2026-11-16, reviewed (plain channel kinetics: no `VERBATIM` C blocks,
includes or system calls) and are pinned here so the app never compiles
code fetched at runtime.

Most kinetics originate from Hay et al. (2011), *PLoS Comput Biol* 7(7):e1002107
(Blue Brain Project); each file names its original reference in its header.

Use of this data is subject to the Allen Institute's terms of use:
https://alleninstitute.org/terms-of-use/ . Citation:
Allen Institute for Brain Science (2015). Allen Cell Types Database.
Available from celltypes.brain-map.org. Technical white paper:
"Biophysical modeling – perisomatic".

`neurosim.channels.ensure_perisomatic_channels()` copies these files to
`~/.cache/neurosim/mechanisms/`, compiles them with NEURON's `nrnivmodl`
(requires a C/C++ compiler) and loads the library into NEURON.

## SHA-256 checksums

```
f012def9758f7d0244b81c54291edea6597c15fad99b9f7b5725b121a3567f16  CaDynamics.mod
021d34cf9ef3e76b0f2afee1a1eeab40eed56d0d3b1362b7e1019f4433273c71  Ca_HVA.mod
a7a9ef262541dc96f5d19afd97778f9320808d01ce397350f1f2be3db30f920f  Ca_LVA.mod
8616325d83add9a1a6546972352b4a62e57095ba25f3c6b5a57ab8b397e2d662  Ih.mod
e1c3bfb2186439bdb963ff2c53268efb824d8e00401aef0a14b7d2234071bb2c  Im.mod
bc378428d3b4da76b6f9b1820dde1f52abef47389ce6edbcddb0c8fb967970fd  Im_v2.mod
2254697617c8da162ff8e5495e4d512fce53f77b44b86754855f11cee4ab3b3e  K_P.mod
1cb9d72f985ef028922cafafed079df5c47b87a893001304181ec48881367ba7  K_T.mod
1cac92486bf77c2976cad9a34e1741dc20ecda70d7b2c9bf5c3474e66f7dfcce  Kd.mod
9244b0eb301dacee5a844910d6b4945856b90dc60937a74ac7f4f19f778d0895  Kv2like.mod
e636c3437a9d9ccb69a0a7efc09fcfe1f16ad97f0468615b70fdbc352fe0855c  Kv3_1.mod
fd6af6564b365f5ec016b453c8f540a15780dea62597053d10cc402443541caf  NaTa.mod
d75ed63d6367fe27c6e5671704594a510e8d83b57db835f30fc36e2f24bef6cf  NaTs.mod
635605ba861c070f8fdc273365f3a0fca6eee24762afc9b102161fdd6e9cfc42  NaV.mod
979d21508f5b089e7b782835ea7f60311223bdc3a8e5242056c594305d2fb220  Nap.mod
079f1e3cac7444696d480b8e02853b215cd93e0c7b9986dbb54925fc49eddbeb  SK.mod
```
