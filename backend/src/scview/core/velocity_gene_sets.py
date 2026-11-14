"""Built-in gene lists a velocity run can leave out (``core/velocity.py`` ``exclude_sets``).

MURK genes (multiple rate kinetics): Barile et al. 2021, Genome Biology 22:197
(doi:10.1186/s13059-021-02414-y, PMC8258993), Additional file 3 (Table S2), "List of mouse MURK
genes identified in Fig. 3B-C, ranked by calculated increase in slope value": 89 mouse genes whose
transcription steps up in Erythroid3 of the Pijuan-Sala 2019 gastrulation atlas, so scVelo reads
the still-rising Erythroid1/2 cells as switching them off and runs the branch backwards. Removing
them restored forward velocity there. Mouse symbols, matched to the counts by name. Downloaded
2026-11-14 from SOURCE_URL (3,454 bytes, sha256 SOURCE_SHA256); kept in the paper's order.
"""

from __future__ import annotations

MURK_BARILE_2021 = (
    'Hba-x', 'Smim1', 'Dcxr', 'Cox6b2', 'Gypa', 'Cpox', 'Hbb-bh1', 'Nudt4', 'Hbb-bt', 'Alad', 'Gpx1', 'Hba-a2',
    'Slc25a21', 'Klf3', 'Hsd3b6', 'Tfrc', 'Mgst3', 'Tmem14c', 'Ftl1', 'Arf5', 'Ell2', 'Hbb-y', 'Fech', 'Ssbp2',
    'Grap2', 'Rhag', 'Fam210b', 'Arid3a', 'Rfesd', 'Bpgm', 'Alas2', 'Abcg2', 'Hesx1', 'Hemgn', 'Pim1', 'Dhrs11',
    'Chchd10', 'Txlng', 'Rmnd5b', 'Prdx2', 'Klhl26', 'Slc4a1', 'Blvrb', 'Abcb10', 'Hebp1', 'Ucp2', 'Mllt3', 'Pklr',
    'Gclm', 'Aqp8', 'Lpcat1', 'Top1mt', 'Cdkn3', 'Gstm5', 'Cdc25b', 'Creg1', 'Dnaja4', 'Kcnn4', 'Slc30a10', 'Svbp',
    'Tmod1', 'Klf1', 'Cdc42ep3', 'Celf4', 'Prr13', 'Arrb1', 'Cldn13', 'Glrx5', 'Ncf4', 'Ppox', 'Rragd', 'Ctsb',
    'Ddias', 'Aqp3', 'Acp1', 'Ddc', 'Gadd45a', 'Nxpe2', 'Fam126a', 'Efcab2', 'Ypel3', 'Hbq1b', 'Mindy2', 'Icam4',
    'Fuca2', 'Kdm7a', 'Htatip2', 'Asns', 'Slc25a37',
)
SOURCE_URL = ("https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-021-02414-y/MediaObjects/"
              "13059_2021_2414_MOESM3_ESM.csv")
SOURCE_SHA256 = "26483df527df7658712dd55bc515670b437fa2a4e12e6f7218353c5b3f357585"

GENE_SETS = {
    "murk_barile2021": {
        "genes": MURK_BARILE_2021,
        "name": "MURK genes (Barile 2021)",
        "species": "mouse",
        "source": "Barile et al. 2021, Genome Biol 22:197, Additional file 3 (Table S2)",
        "url": SOURCE_URL,
        "sha256": SOURCE_SHA256,
        "note": ("89 mouse erythroid genes whose transcription steps up late (Erythroid3 of the gastrulation atlas); "
                 "they turn scVelo's erythroid flow backwards there. Meant for mouse erythroid data."),
    },
}


def describe() -> list[dict]:
    """The built-in lists for the panel: id, name, size, source and when to use it."""
    return [{"id": k, "name": v["name"], "n_genes": len(v["genes"]), "species": v["species"], "source": v["source"],
             "url": v["url"], "note": v["note"]} for k, v in GENE_SETS.items()]
