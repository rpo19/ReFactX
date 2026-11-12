# Piano operativo

## 0. Obiettivo

### Goal: mostrare le capacità distintive del sistema

**Controllo + Power of the Reasoner**

* Dimostrare performance migliori su:

  * **count queries**
  * **multi-hop queries**
  * **superlativi (optional)**
* Dimostrare la capacità di ottenere **conoscenza derivata in real time**, ad esempio:

  * `control`
  * `ultimate_control`

**Flessibilità**

* Mostrare la possibilità di **scrivere e aggiungere nuove regole** in modo flessibile.

### In generale

* Mostrare una **explainable trace** del processo di reasoning, dalla query ai risultati.

---

# 1. Dataset / Data

### Dataset principale

* Utilizzare il dataset relativo al **company ownership** del paper precedente.

### Alternative / benchmark pubblici

* **Wikidata** (~800M triple) o un altro database pubblico.

  * Verificare in particolare quali **regole/relazioni derivate** siano già disponibili o possano essere definite.
* **YAGO 4.5**

  * Valutare l'utilizzo di YAGO 4.5 come dataset aggiuntivo.
  * Riferimento: https://doi.org/10.1145/3626772.3657876
  * **TODO:** verificare il dataset e la sua adeguatezza al benchmark.

---

# 2. Confronto / Benchmark

L'obiettivo principale è confrontare il sistema con i risultati ottenuti da **RefactX** sul dataset di company ownership del paper precedente.

Particolare attenzione a:

* **count queries**
* **derived knowledge**

  * `control`
  * `ultimate_control`
* **(optional superlativi)**
* **multi-hop reasoning**

### Query del vecchio database

Query generiche:

* `list_query`
* `yes_no_query`

Query più complesse:

* `count_query`
* `multi_hop`

Mantenere, come ordine di grandezza, un limite di circa **10 triple** per le query / risposte del benchmark.

### Note implementative

* In caso di risposte molto lunghe:

  * restituire direttamente la **query vadalog**, oppure
  * restituire direttamente la **risposta**.
* In Vadalog, `control` e `ultimate_controller` **non sono materializzati**: vengono derivati dal reasoning.
* **TODO:** aggiungere query con **superlativi**.

---

# 3. Architettura / Interazione LLM ↔ Vadalog

L'idea è separare il processo in tre livelli di interazione.

### 3.1 Prefix → indice delle entità

Il **prefix** contiene l'indice dei nomi delle entità disponibili, senza includere tutte le triple.

Questo permette di vincolare la generazione dell'LLM alle entità effettivamente presenti nel dataset.

---

### 3.2 Proprietà → relazione + regole applicabili

Per una determinata entità, Vadalog restituisce il **set di proprietà/relazioni disponibili**, ad esempio:

* `control`
* `own`
* `ultimate_control`
* `count_controllate`
* ...

Queste proprietà sono determinate dinamicamente da Vadalog sulla base dell'entità corrente e delle regole disponibili.

Una volta ottenuti:

**relation + regole → get results**

l'LLM può scegliere quale proprietà utilizzare nella query.

---

### 3.3 Generazione della query

Flusso indicativo:

**1. LLM → inizio della query**

L'LLM inizia a generare la query.

Quando genera:

> `Fact:`

si attiva la **prima interazione con PrefixTree**, che permette di vincolare il set di token generabili rispetto alle entità effettivamente esistenti nel database.

**2. LLM → generazione dell'entità**

L'LLM continua la generazione e identifica l'entità.

A questo punto avviene la **seconda interazione con Vadalog**:

→ Vadalog restituisce il **set di proprietà disponibili per quella specifica entità**.

L'LLM seleziona quindi la property/relation che vuole utilizzare e continua a generare la query.

**3. LLM → richiesta dei risultati**

Viene effettuata la **terza interazione con Vadalog**:

→ Vadalog esegue il reasoning e restituisce i risultati, idealmente come **stream di dati**.

In questo modo l'LLM non deve necessariamente generare o ricostruire tutto il contenuto del database: può interrogare dinamicamente il reasoner e utilizzare i risultati prodotti.

---

# TODO — AC

* Ottenere un **semplice dataset sintetico** composto da triple.
* Preparare uno **setup/demo delle API di Vadalog**.

---

# Looking forward

### 1. Rule generation from scratch

Andare oltre un set di proprietà predefinite:

* permettere all'LLM di **scrivere nuove regole da zero**;
* utilizzare il reasoner per verificare/eseguire tali regole;
* eliminare progressivamente la dipendenza da un insieme fisso di proprietà predefinite.

### 2. Training degli LLM

Come direzione successiva, valutare il **training degli LLM** per sfruttare nativamente questo paradigma di interazione con il reasoner:

* generazione guidata dalle entità disponibili;
* discovery delle proprietà;
* interazione iterativa con il reasoner;
* utilizzo della conoscenza derivata in real time.
