#!/usr/bin/env bash
# Download all public data used by VirtualCell-Perturb into data/raw/.
#   Perturb-seq (GEARS-processed, Harvard Dataverse):
#     - Replogle et al. 2022, K562 essential-gene CRISPRi screen
#     - Adamson et al. 2016, K562 UPR CRISPRi screen
#   Prior knowledge:
#     - GO annotations (GEARS gene2go.pkl), GO ontology (go-basic.obo)
#     - STRING v12 human PPI network
#     - Enrichr gene-set libraries (Hallmark, GO BP, KEGG, Reactome, CORUM)
#     - GenePT gene embeddings (Zenodo 10833191; optional, 574 MB, parallel range download)
set -e
cd "$(dirname "$0")/../data/raw"
DV=https://dataverse.harvard.edu/api/access/datafile

get() { [ -s "$2" ] || curl -sL --retry 5 -o "$2" "$1"; echo "ok  $2"; }

get $DV/7458695 replogle_k562_essential.zip
get $DV/6154417 adamson.zip
get $DV/6153417 gene2go.pkl
for z in replogle_k562_essential adamson; do [ -d $z ] || unzip -q $z.zip; done

mkdir -p string genesets
get https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz string/9606.protein.links.v12.0.txt.gz
get https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz string/9606.protein.info.v12.0.txt.gz
get https://purl.obolibrary.org/obo/go/go-basic.obo go-basic.obo
for lib in MSigDB_Hallmark_2020 GO_Biological_Process_2023 KEGG_2021_Human Reactome_2022 CORUM; do
  get "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=$lib" genesets/$lib.gmt
done

if [ "${SKIP_GENEPT:-0}" != "1" ] && [ ! -s GenePT_emebdding_v2.zip.done ]; then
  URL="https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1"
  SIZE=574395233; N=${N:-8}; CH=$(( (SIZE + N - 1) / N ))
  mkdir -p genept_parts
  for i in $(seq 0 $((N-1))); do
    S=$((i*CH)); E=$(( (i+1)*CH - 1 )); [ $E -ge $SIZE ] && E=$((SIZE-1))
    curl -sL --retry 20 -r $S-$E -o genept_parts/part$i "$URL" &
  done
  wait
  cat $(for i in $(seq 0 $((N-1))); do echo genept_parts/part$i; done) > GenePT_emebdding_v2.zip
  unzip -q -o GenePT_emebdding_v2.zip -d GenePT && rm -rf genept_parts && echo done > GenePT_emebdding_v2.zip.done
fi
echo "all data ready"
