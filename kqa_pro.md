# KQA Pro: preparazione dataset, indice e valutazione

Questo documento riassume cosa è stato fatto per usare **KQA Pro** con ReFactX:
download, verifica dei riferimenti al dump Wikidata, verbalizzazione,
costruzione dell'indice, codice di valutazione, analisi di copertura e idee per
il training.

---

## 1. Verifica del dump Wikidata

KQA Pro **non viene ricostruito da un dump Wikidata completo** al momento
dell'indicizzazione. Il dataset distribuisce direttamente una KB densa,
`kb.json`, costruita partendo dalle entità di **FB15k-237** allineate a
Wikidata.

Nel paper e nel repository **non è indicato un dump Wikidata specifico** con
data o versione. Il paper dice soltanto che le entità sono state allineate a
Wikidata per estrarre attributi, relazioni e qualifier.

Link ufficiali verificati:

| Risorsa | Stato |
|---|---|
| `cloud.tsinghua.edu.cn/f/04ce81541e704a648b03` | link morto: *"Link does not exist."* |
| homepage `thukeg.gitee.io/kqa-pro` | **404** |
| mirror HuggingFace `drt/kqa_pro` | funzionante |
| mirror HuggingFace `caobiao24/kqa_pro` | funzionante |
| mirror HuggingFace `soongfs/kqa_pro` | funzionante |

I tre mirror contengono gli stessi file e le stesse dimensioni:

```text
kb.json      ~79 MB
train.json   ~88 MB
val.json     ~11 MB
test.json    ~3.2 MB
```

**Conclusione:** non esiste un dump completo scaricabile collegato a KQA Pro.
Non è quindi possibile/utile indicizzare “l'intero dump Wikidata”; è stata
indicizzata l'intera KB ufficiale di KQA Pro.

---

## 2. Download

I file sono stati salvati in `/workspace/data/kqa_pro`:

```bash
mkdir -p /workspace/data/kqa_pro
cd /workspace/data/kqa_pro
for f in kb.json train.json val.json test.json README.md; do
  curl -L --retry 3 -o "$f" \
    "https://huggingface.co/datasets/drt/kqa_pro/resolve/main/$f?download=true"
done
```

---

## 3. Statistiche dataset

| Split | Domande |
|---|---:|
| train | **94.376** (94.364 uniche) |
| val | **11.797** |
| test | **11.797** |

KB:

| Elemento | Numero |
|---|---:|
| concetti | 794 |
| entità | 16.960 |
| nomi totali | 17.754 |
| nomi duplicati | 1.819 |
| `instance of` | 17.325 |
| attributi | 174.539 |
| relazioni | 415.334 |
| triple base uniche | **376.318** |
| triple con qualifier | **534.346** |

Distribuzione train:

| Categoria | Esempi | Percentuale |
|---|---:|---:|
| Domande con qualifier | **21.629** | 22,9% |
| Domande con `Count` | **10.946** | 11,6% |
| Domande `Verify*` (yes/no) | **11.653** | 12,3% |
| Almeno 2 `Relate` (multi-hop) | **8.281** | 8,8% |
| Con almeno un `Relate` | **41.148** | 43,6% |

Tipi di risposta in train:

| Tipo | Esempi |
|---|---:|
| string/entity | 58.556 |
| integer | 17.811 |
| yes/no | 11.655 |
| date | 6.354 |

Lunghezza media del programma KoPL: **4,78** passi; massimo **15**.

---

## 4. Codice aggiunto

| File | Scopo |
|---|---|
| `utils/prep_kqa_pro.py` | verbalizza `kb.json`, disambigua nomi duplicati e converte gli split in JSONL |
| `utils/build_kqa_pro_index.py` | costruisce indice Postgres oppure indice locale `.pkl` |
| `prompts/prompt_kqa_pro.yaml` | prompt fact/count per la KB base |
| `prompts/prompt_kqa_pro_qualifiers.yaml` | prompt con esempi di qualifier |
| `configs/config_kqa_pro_val_qwen36_27b.json` | eval val con indice Postgres |
| `configs/config_kqa_pro_val_qwen36_27b_local.json` | eval val con indice locale base |
| `configs/config_kqa_pro_val_qwen36_27b_qualifiers.json` | eval val con indice locale + qualifier |
| `refactx/index.py` | aggiunto `LocalIndex`, salvataggio/caricamento `.pkl`, supporto `load_index("*.pkl")` |
| `utils/eval.py` | supporto `data_files`, `choices_key` e `index_path` locale |

---

## 5. Verbalizzazione

Il formato usato è coerente con gli altri indici ReFactX:

```text
<subject> <predicate> <object> .
```

Con prefisso `" "` al momento dell'indicizzazione e terminatore `" ."`.

Regole principali:

- `instanceOf` → relazione `instance of`
- relazioni forward: `<entity> <predicate> <object>`
- relazioni backward: `<object> <predicate> <entity>`
- attributi → `<entity> <key> <value>`
- valori:
  - string → testo normalizzato
  - quantity → numero + unità, omettendo l'unità `1`
  - year → anno a 4 cifre
  - date → `YYYY-MM-DD`
- nomi duplicati → `Nome (Qid)`, es. `Kettering (Q984613)`
- deduplica finale delle triple

Comando:

```bash
python utils/prep_kqa_pro.py \
  --data-dir /workspace/data/kqa_pro \
  --out-dir /workspace/data/kqa_pro
```

Output principali:

```text
/workspace/data/kqa_pro/kqa_pro_triples.txt.gz
/workspace/data/kqa_pro/train.jsonl
/workspace/data/kqa_pro/val.jsonl
/workspace/data/kqa_pro/test.jsonl
/workspace/data/kqa_pro/kqa_pro_stats.json
```

---

## 6. Qualifier

I qualifier non sono triplette standard ReFactX. È stata aggiunta una
rappresentazione sperimentale, opzionale:

```text
<Subject> <predicate (qualifier key)> <value> .
```

Esempi:

```text
<Gregg Allman> <spouse> <Cher (Q12003)> .
<Gregg Allman> <spouse (start time)> <1975> .
<Gregg Allman> <spouse (end time)> <1979> .

<20th Century Fox> <owned by> <News Corporation> .
<20th Century Fox> <owned by (start time)> <1984> .
<20th Century Fox> <owned by (end time)> <2013> .

<A Beautiful Mind> <nominated for> <Academy Award for Best Supporting Actress> .
<A Beautiful Mind> <nominated for (statement is subject of)> <74th Academy Awards> .
<A Beautiful Mind> <nominated for (nominee)> <Jennifer Connelly> .
```

Comando per generare anche i qualifier:

```bash
python utils/prep_kqa_pro.py \
  --data-dir /workspace/data/kqa_pro \
  --out-dir /workspace/data/kqa_pro/qualifiers \
  --include-qualifiers \
  --no-write-splits \
  --triples-name kqa_pro_triples_qualifiers.txt.gz
```

Limite importante: il qualifier è agganciato a `<Subject> <predicate>`, **non**
alla specifica coppia `<Subject> <predicate> <Object>`. Se lo stesso subject ha
più oggetti per la stessa relazione, il qualifier può essere ambiguo. Per questo
l'indice base non include i qualifier e l'indice qualifier è opzionale.

---

## 7. Indice

### Indice locale

Il backend locale salva un `LocalIndex` in pickle ed è compatibile con il
decoder vincolato:

```bash
python utils/build_kqa_pro_index.py \
  /workspace/data/kqa_pro/kqa_pro_triples.txt.gz \
  --backend local \
  --output /workspace/data/kqa_pro/kqapro_qwen36.pkl \
  --model-name Qwen/Qwen3.6-27B
```

Indice con qualifier:

```bash
python utils/build_kqa_pro_index.py \
  /workspace/data/kqa_pro/qualifiers/kqa_pro_triples_qualifiers.txt.gz \
  --backend local \
  --output /workspace/data/kqa_pro/kqapro_qwen36_qualifiers.pkl \
  --model-name Qwen/Qwen3.6-27B
```

Output creati:

| File | Contenuto |
|---|---|
| `kqa_pro_triples.txt.gz` | 376.318 triple base uniche |
| `kqapro_qwen36.pkl` | indice locale base, ~15 MB |
| `kqa_pro_triples_qualifiers.txt.gz` | 534.346 triple con qualifier |
| `kqapro_qwen36_qualifiers.pkl` | indice locale con qualifier, ~20 MB |

L'indice pieno è stato caricato e verificato: `root leaves = 376318`.

### Indice Postgres

Il comando previsto è:

```bash
python utils/build_kqa_pro_index.py \
  /workspace/data/kqa_pro/kqa_pro_triples.txt.gz \
  --backend postgres \
  --table-name kqapro_qwen36 \
  --model-name Qwen/Qwen3.6-27B
```

**Nota:** l'utente Postgres `secondment` non ha privilegi `CREATE` sullo schema
`public`, quindi non è stato possibile creare una nuova tabella persistente. La
tabella `qwen36` non è stata toccata. Per questo è stato usato il backend locale
`.pkl`. Se viene fornito un ruolo con permessi di creazione, il comando sopra è
pronto.

---

## 8. Test di verbalizzazione e indicizzazione

Smoke test su 5.000 triple:

```text
<(500) Days of Summer> <cast member> <Zooey Deschanel> .
```

Count tool verificato sull'indice pieno:

```text
<(500) Days of Summer> <cast member>          -> 3
<Kettering (Q984613)> <instance of>           -> 1
<LeBron James> <award received>               -> 2
<LeBron James> <drafted by>                   -> 1
```

Test esistenti:

- `tests/test_features.py` su `CountBranches` / `FactGeneration`: **20 passed**
- `tests/test_routed_index.py` + `tests/test_efficiency.py`: **17 passed**

---

## 9. Valutazione

`utils/eval.py` è stato esteso per supportare:

- `data_files`: caricamento di JSONL locale via `datasets`
- `choices_key`: aggiunta delle 10 scelte al prompt
- `index_path`: indice locale `.pkl` oltre a Postgres/HTTP

Eval con indice locale base:

```bash
python utils/eval.py \
  --config configs/config_kqa_pro_val_qwen36_27b_local.json
```

Eval con indice locale + qualifier:

```bash
python utils/eval.py \
  --config configs/config_kqa_pro_val_qwen36_27b_qualifiers.json
```

Eval con Postgres, dopo aver creato la tabella:

```bash
python utils/eval.py \
  --config configs/config_kqa_pro_val_qwen36_27b.json
```

L'eval non è stata lanciata perché richiede la GPU, già occupata da un altro
job.

### Copertura val

| Metrica | Valore |
|---|---:|
| Input `Find` presenti nell'indice | 16.058 / 16.058 (100%) |
| Domande con tutti gli input `Find` presenti | 8.814 / 11.797 (74,7%) |
| Gold answer presente come entità indicizzata | 4.928 / 11.797 (41,8%) |
| Domande con passi qualifier | 2.698 / 11.797 (22,9%) |

Il resto delle gold answer sono valori letterali, date, numeri o yes/no.

---

## 10. KQA Pro è adatto al training?

**Sì, soprattutto per SFT di tool use e reasoning su KB.** È più utile di molti
altri dataset perché fornisce:

- gold program KoPL
- gold SPARQL
- answer verificabile
- domande multi-hop, count, comparison, logical e qualifier
- split train/val/test già separati

È quindi possibile generare trace verificate con `<fact>` e `<count>`, non solo
usare il dataset per la valutazione.

### Ha abbastanza esempi?

Sì:

- **94.376** domande train
- **21.629** domande train con qualifier
- **10.946** domande train con `Count`
- **11.653** domande train `Verify*`
- **8.281** domande train con almeno 2 `Relate`

Il sottoinsieme qualifier da solo è più grande di molti benchmark usati per il
training. 94k esempi sono più che sufficienti per SFT.

### Caveat

- **KB piccola:** circa 17.000 entità e 376.000 triple base. Il modello può
  memorizzare parte della KB invece di imparare una procedura generale.
- **Domande sintetiche+parafrasate:** 94k uniche, ma con diversità linguistica
  inferiore a dataset interamente scritti da umani.
- **Choices:** ogni domanda ha 10 scelte. Se vengono sempre passate al modello,
  il modello può imparare a scegliere invece di ragionare. Consiglio di
  addestrare senza choices, oppure con choices rimescolate, e valutare entrambe
  le impostazioni.
- **Test set senza risposte:** usare `val` per eval locale; `test` solo per
  predizioni.
- **Qualifier custom:** rappresentazione non standard. Se si addestra sui
  qualifier, usare coerentemente indice e prompt qualifier.
- **Ambiguità qualifier:** filtrare o marcare i casi con più oggetti per la
  stessa relazione.

---

## 11. Idee per il training

### Mix consigliato

| Componente | Quota indicativa |
|---|---:|
| KQA Pro base | 60–70% |
| KQA Pro qualifier | 15–20% |
| KQA Pro count / verify / multi-hop | 15–20% |
| Dataset già usati (Mintaka, 2Wiki, WebQSP, CWQ) | 20–40% |

KQA Pro da solo insegna il formato e il tool use, ma non basta per generalizzare
su KB grandi e domini aperti. Va mischiato con i dataset già in uso.

### Costruzione delle trace SFT

1. Eseguire o validare il programma KoPL/SPARQL contro `kb.json`.
2. Estrarre i fatti necessari dal programma.
3. Verificare che ogni fatto esista nell'indice ReFactX.
4. Per i count, generare `<count> <S> <R>` e confrontare il valore con
   l'esecuzione del programma.
5. Per i qualifier, verificare che esistano le triple
   `<S> <R (qualifier key)> <value>`.
6. Scartare o riparare gli esempi in cui i fatti richiesti sono assenti o
   ambigui.
7. Opzionale: aggiungere trace negative con `<no further records>`.

### Curriculum suggerito

1. Domande a 1–2 hop senza qualifier.
2. Domande con count.
3. Domande con comparison e logical.
4. Domande con qualifier.
5. Multi-hop con qualifier.

### Scelte multiple

Per evitare shortcut:

- training: nascondere le choices e addestrare a generare la risposta;
- eval: riportare sia accuracy con choices sia accuracy senza choices;
- opzionale: addestrare anche un setting “answer selection” separato.

### RL / GRPO

KQA Pro può essere usato anche per RL perché la risposta è verificabile:

- reward esatto sulla risposta;
- reward intermedio sul fatto/count corretto;
- penalità per tool call invalide o loop;
- filtrare esempi con qualifier ambigui.

Con 94k esempi, il dataset è abbastanza grande anche per RL, ma conviene
campionare un sottoinsieme bilanciato per categoria.

---

## 12. Comandi rapidi

```bash
# 1. Verbalizzazione + JSONL
python utils/prep_kqa_pro.py \
  --data-dir /workspace/data/kqa_pro \
  --out-dir /workspace/data/kqa_pro

# 2. Indice locale base
python utils/build_kqa_pro_index.py \
  /workspace/data/kqa_pro/kqa_pro_triples.txt.gz \
  --backend local \
  --output /workspace/data/kqa_pro/kqapro_qwen36.pkl

# 3. Indice con qualifier
python utils/prep_kqa_pro.py \
  --data-dir /workspace/data/kqa_pro \
  --out-dir /workspace/data/kqa_pro/qualifiers \
  --include-qualifiers \
  --no-write-splits \
  --triples-name kqa_pro_triples_qualifiers.txt.gz

python utils/build_kqa_pro_index.py \
  /workspace/data/kqa_pro/qualifiers/kqa_pro_triples_qualifiers.txt.gz \
  --backend local \
  --output /workspace/data/kqa_pro/kqapro_qwen36_qualifiers.pkl

# 4. Eval
python utils/eval.py \
  --config configs/config_kqa_pro_val_qwen36_27b_local.json

python utils/eval.py \
  --config configs/config_kqa_pro_val_qwen36_27b_qualifiers.json
```

---

## 13. Pipeline HPC: esempi di training con Qwen 27B

Obiettivo: sostituire Mintaka con KQA Pro per il training dei **count** e per lo
studio dei **tipi di domanda**, generando le trace con il teacher Qwen3.8-27B.

Script aggiunti:

| File | Scopo |
|---|---|
| `utils/sample_kqa_pro.py` | campiona un sottoinsieme bilanciato di train (count / verify / multi-hop / qualifier / base) |
| `utils/build_kqa_sft.py` | converte i log teacher in trace SFT, filtra `evaluation.correct==1`, ripara i tool call, aggiunge EOS e metadati di tipo |
| `utils/analyze_kqa_types.py` | report di accuracy/truncation/tool use per tipo di programma KoPL e per tipo di risposta |
| `utils/mix_kqa_sft.py` | mixa KQA Pro nella mixture pulita e rimuove i count Mintaka |
| `hpc/prep_kqa_pro.sh` | scarica i file, costruisce indice locale `.pkl` con tokenizer Qwen3.8-27B |
| `hpc/eval_kqa_pro_train_qwen38_27b.sh` | genera esempi teacher sul train (4 GPU) |
| `hpc/eval_kqa_pro_val_qwen38_27b.sh` | eval teacher su val (4 GPU, 1000 esempi) |
| `hpc/build_kqa_sft.sh` | analisi tipi + build SFT + mix finale |
| `hpc/sft_train_kqa_08b.sh` | training 0.8B sulla mixture KQA Pro (2 GPU) |

### Ordine di esecuzione

```bash
# 1. una tantum: dati + indice locale
sbatch hpc/prep_kqa_pro.sh

# 2. teacher Qwen3.8-27B sul train (12k esempi bilanciati di default)
sbatch hpc/eval_kqa_pro_train_qwen38_27b.sh

# 3. analisi tipi + costruzione SFT + mix (Mintaka count rimosso)
sbatch hpc/build_kqa_sft.sh

# 4. training 0.8B
sbatch hpc/sft_train_kqa_08b.sh
```

Variabili utili:

```bash
# directory dati/indice (default: $WS_PATH/kqa_pro)
KQAPRO_DIR=/data/.../kqa_pro sbatch hpc/eval_kqa_pro_train_qwen38_27b.sh
# numero di esempi teacher (default 12000)
KQAPRO_SAMPLE_TOTAL=20000 sbatch hpc/eval_kqa_pro_train_qwen38_27b.sh
# log teacher alternativo
KQA_LOG=logs/kqa_pro_train_eval_qwen38_27b.out.0 sbatch hpc/build_kqa_sft.sh
```

Output principali:

```text
$WS_PATH/kqa_pro/kqapro_qwen38.pkl
$WS_PATH/kqa_pro/train_sample_ids.json
logs/kqa_pro_train_eval_qwen38_27b.out.0
logs/kqa_pro_type_report.md
logs/kqa_pro_sft.jsonl
logs/sft_training_mix_kqa.jsonl
```

### Note importanti

- Le **choices non vengono passate** al teacher: servono per l'eval diagnostica,
  non per il training, altrimenti il modello impara a scegliere tra 10 opzioni.
- L'indice `.pkl` va costruito con il tokenizer del modello usato in eval
  (`Qwen3.8-27B`). I tokenizer Qwen3.5/3.6/3.8 condividono lo stesso vocabolario
  (verificato localmente su 3.5-0.8B e 3.6-27B), quindi lo stesso indice resta
  compatibile con lo student 0.8B.
- `utils/mix_kqa_sft.py` rimuove di default i record Mintaka con
  `complexityType/question_type == count` e aggiunge KQA Pro. Usa
  `--keep-mintaka-count` per non rimuoverli.
- Per lo studio dei tipi, `utils/analyze_kqa_types.py` produce
  `logs/kqa_pro_type_report.md` e `.json` con accuratezza per tipo KoPL e per
  tipo di risposta.
