# Rezumat Paper BusterX++

**Paper:** BusterX++: Towards Unified Cross-Modal AI-Generated Content Detection and Explanation with MLLM
**arXiv:** 2507.14632
**Scop:** Acest document rezumă fiecare secțiune a lucrării pentru referința echipei.

---

## 3.2 Construcția Benchmark-ului

### Prezentare Generală

| Atribut | Valoare |
|---------|---------|
| Total eșantioane | 4.000 |
| Imagini reale | 1.000 |
| Imagini false | 1.000 |
| Videoclipuri reale | 1.000 |
| Videoclipuri false | 1.000 |

---

### Surse de Date

#### Conținut Real
- **Sursă:** Dataset-ul OpenVid-1M HD
- **Conținut:** Scenarii diverse din lumea reală
- **Pre-filtrare:** Aplicată pentru varietate de scene

#### Conținut Fals
Două surse:

1. **MagicArena** - eșantioane curate cu rating ridicat

2. **Pipeline de Generare Custom:**
```
Reddit API (imagini reale)
    → Qwen-2.5-VL (generează descrieri/captions)
    → Descrierile devin prompturi
    → Modelele diffusion generează conținut fals
```

**De ce Reddit?** Imaginile reale de pe social media oferă scenarii realiste. Descrierile captează situații autentice, făcând conținutul generat mai greu de detectat.

| Tip Generator | Modele Folosite |
|---------------|-----------------|
| Imagini | FLUX, GPT-4o |
| Videoclipuri | Seedance 1.0, SkyReels V1 |

---

### Filtrarea Datelor

#### Eșantioane Reale (3 etape)
1. **Filtru tehnic:** rezoluție, frame rate, bitrate
2. **Eliminare duplicate:** din același clip original
3. **Verificare manuală:** eliminare watermark-uri, anime, fundaluri sintetice evidente

#### Eșantioane False (Abordare nouă în 2 etape)
1. Amestecă real + fals într-un pool orb → experții identifică eșantioanele care "par reale"
2. Re-examinare pentru confirmarea originii sintetice

**Scopul:** Păstrează doar cele mai convingătoare falsuri (benchmark provocator).

---

### Specificații Post-Procesare

| Media | Parametru | Valoare |
|-------|-----------|---------|
| **Imagini** | Rezoluție | 1024 × 1024 |
| **Videoclipuri** | Rezoluție | 1920 × 1080 |
| | Durată | 5 secunde |
| | Frame rate | 24 FPS |
| | Codec | HEVC x265 |

**De ce standardizăm?**
1. Elimină bias-urile de encoding între diferiți generatori
2. Asigură consistență între toate sursele

#### Comenzi FFmpeg
```bash
# Video
ffmpeg -i input.mp4 \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" \
  -t 5 -r 24 -c:v libx265 -pix_fmt yuv420p10le output.mp4

# Imagine
ffmpeg -i input.jpg \
  -vf "scale=1024:1024:force_original_aspect_ratio=decrease,pad=1024:1024:(ow-iw)/2:(oh-ih)/2" \
  output.jpg
```

---

### Referințe Folosite în Această Secțiune

| Ref | Nume | Scop |
|-----|------|------|
| [5] | Qwen-2.5-VL | Captioning imagini |
| [17] | Seedance 1.0 | Generare video |
| [29] | FLUX | Generare imagini |
| [33] | OpenVid-1M HD | Sursă date reale |
| [36] | GPT-4o | Generare imagini |
| [43] | SkyReels V1 | Generare video |

---

## 4. Metodă

### Cum Se Leagă de Construcția Benchmark-ului

După pregătirea dataset-ului (Secțiunea 3.2), următorul pas este antrenarea. Fluxul este:

```
Benchmark (4.000 eșantioane)
    → Post-procesat (rezoluție, durată, codec standardizate)
    → Introdus în pipeline-ul de antrenare
    → Modelul învață să clasifice Real vs Fals
```

---

## 4.1 Provocările Cold Start

### Ce Este Cold Start?

Majoritatea metodelor MLLM+RL folosesc o abordare în două faze:
1. **Faza SFT (Cold Start):** Supervised fine-tuning cu exemple Chain-of-Thought (CoT)
2. **Faza RL:** Reinforcement learning pentru îmbunătățire

### Problema

Autorii argumentează că această abordare este **limitată** deoarece:

| Problemă | Explicație |
|----------|------------|
| **Blocaj în calitatea CoT** | Detectarea umană a falsurilor se bazează pe indicii subtile, intuitive (reflexii nenaturale, inconsistențe de lumină, artefacte de mișcare) |
| **Greu de generat CoT bun** | Crearea explicațiilor de calitate prin prompt engineering este extrem de dificilă |
| **Risc de degradare** | Datele CoT de calitate slabă pot degrada capacitatea de raționament a modelului |

### Soluția BusterX++

**Abandonează complet cold-start-ul.** În schimb, folosește o abordare RL multi-etape care construiește capabilitățile progresiv, fără a necesita date CoT pre-generate.

---

## 4.2 Antrenare Multi-Etape

### Prezentare Generală

BusterX++ folosește **DAPO** (Dynamic sAmpling Policy Optimization) în trei etape:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FLUXUL PIPELINE-ULUI DE ANTRENARE               │
└─────────────────────────────────────────────────────────────────────┘

    Dataset Standardizat (din Secțiunea 3.2)
                    │
                    ▼
    ┌───────────────────────────────────┐
    │   ETAPA 1: Foundation RL          │  ~70% din antrenare
    │   Învață clasificare de bază      │
    │   Reward: format + lungime + acc  │
    └───────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │   ETAPA 2: Thinking Mode Fusion   │  ~5% din antrenare
    │   Învață să schimbe modurile      │
    │   Metodă: Supervised Fine-Tuning  │
    └───────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │   ETAPA 3: Advanced RL            │  ~25% din antrenare
    │   Îmbunătățește calitatea         │
    │   raționamentului                 │
    │   Reward: + hybrid + thinking     │
    └───────────────────────────────────┘
                    │
                    ▼
           Model Final (BusterX++)
```

---

### Ce Este DAPO?

DAPO (Dynamic sAmpling Policy Optimization) este o versiune îmbunătățită a GRPO. Pentru fiecare input:

1. Generează multiple output-uri de la model
2. Scorează fiecare output cu funcții de reward
3. Calculează avantajul (cât de mult mai bun/rău față de medie)
4. Actualizează modelul să favorizeze output-urile cu reward mai mare

**Îmbunătățiri cheie față de GRPO:**
- Clipping asimetric (explorare mai bună)
- Loss la nivel de token (mai bun pentru raționament lung)
- Sampling dinamic (elimină eșantioanele neinformative)

---

### Etapa 1: Foundation RL

**Scop:** Învață clasificarea de bază real vs fals.

**Ce se întâmplă:**
- Modelul primește video/imagine
- Generează răspuns cu format `<think>...</think><answer>...</answer>`
- Primește recompense bazate pe corectitudine

**Funcția de Reward:**
```
R_stage-1 = r_fmt + r_overlong + r_acc
```

| Reward | Valoare | Condiție |
|--------|---------|----------|
| r_fmt | 0 | Format corect (`<think>...</think><answer>...</answer>`) |
| r_fmt | -1 | Format greșit |
| r_overlong | 0 la -1 | Penalizare graduală dacă răspunsul depășește lungimea maximă |
| r_acc | +1 | Clasificare corectă |
| r_acc | 0 | Clasificare greșită |

**Output:** Modelul poate clasifica, dar calitatea raționamentului este de bază.

---

### Etapa 2: Thinking Mode Fusion

**Scop:** Învață modelul să alterneze între raționament detaliat și răspunsuri rapide.

**Metodă:** Supervised Fine-Tuning (NU RL)

**Două moduri:**

| Mod | Declanșator | Format Output |
|-----|-------------|---------------|
| Thinking | `/think` sau fără instrucțiune | `<think>{raționament detaliat}</think><answer>{răspuns}</answer>` |
| Non-Thinking | `/no_think` | `<think></think><answer>{răspuns}</answer>` |

**De ce e necesar?**
- Uneori ai nevoie de explicație detaliată (pentru rapoarte, dovezi)
- Uneori ai nevoie de clasificare rapidă (procesare în batch)
- ~0.7% scădere în acuratețe în modul non-thinking, dar mult mai rapid

**Notă:** Ablația arată că această etapă are impact minim pe acuratețe, dar e necesară pentru Etapa 3.

---

### Etapa 3: Advanced RL

**Scop:** Îmbunătățește calitatea raționamentului folosind evaluare externă.

**Componente noi:**

#### 1. Thinking Reward
Un model extern evaluează calitatea raționamentului:
- **Model:** SophiaVL-R1-Thinking-Reward-Model-3B
- **Scor:** 0 ≤ r_think ≤ 1

```
r_think = {
    0,                      dacă mod /no_think
    min(r_acc, M(y_res)),   altfel (M = model extern)
}
```

**De ce min(r_acc, ...)?** Dacă clasificarea e greșită, raționamentul bun nu ajută.

#### 2. Hybrid Reward
Asigură că modelul respectă instrucțiunile de mod:

```
r_hybrid = {
    0,   dacă răspunsul urmează modul corect
   -1,   dacă gândește când nu trebuie, sau sare peste gândire când trebuie
}
```

**Reward Total:**
```
R_stage-3 = r_fmt + r_overlong + r_acc + r_hybrid + r_think
```

**De ce Etapa 3 e separată:** Aplicarea thinking reward prea devreme (în Etapa 1) destabilizează antrenarea.

---

### Configurație de Antrenare

| Parametru | Valoare |
|-----------|---------|
| Model de Bază | Qwen2.5-VL-7B-Instruct |
| Model Extern de Reward | SophiaVL-R1-Thinking-Reward-Model-3B |
| Metodă Fine-tuning | LoRA (rank=16, alpha=32) |
| Learning Rate | 1×10⁻⁵ |
| Precizie | bfloat16 |

---

### Comparație Performanță

| Abordare | După Etapa 1 | După Etapa 3 |
|----------|--------------|--------------|
| Cu Cold-Start | 71.7% | 72.9% |
| Fără Cold-Start (BusterX++) | 69.4% | **77.4%** |

**Concluzie:** Fără cold-start performează mai slab inițial, dar atinge rezultate finale mai bune cu generalizare superioară.

---

### Flux Complet: De la Date la Model Antrenat

```
1. CONSTRUCȚIE BENCHMARK (Secțiunea 3.2)
   │
   ├── Colectează date reale (OpenVid-1M HD)
   ├── Generează date false (FLUX, GPT-4o, Seedance, SkyReels)
   ├── Filtrează (calitate + review experți)
   └── Standardizează (rezoluție, durată, codec)
   │
   ▼
2. ETAPA 1: FOUNDATION RL
   │
   ├── Input: Imagini/videoclipuri standardizate cu etichete
   ├── Algoritm: DAPO
   ├── Rewards: Format + Lungime + Acuratețe
   └── Output: Clasificator de bază cu raționament
   │
   ▼
3. ETAPA 2: THINKING MODE FUSION
   │
   ├── Input: Output-uri de la modelul din Etapa 1
   ├── Metodă: Supervised Fine-Tuning
   ├── Învață: modurile /think și /no_think
   └── Output: Model cu raționament comutabil
   │
   ▼
4. ETAPA 3: ADVANCED RL
   │
   ├── Input: Aceleași date ca Etapa 1
   ├── Algoritm: DAPO
   ├── Rewards: Rewards Etapa 1 + Hybrid + Thinking (via SophiaVL)
   └── Output: Model final BusterX++
   │
   ▼
5. INFERENȚĂ
   │
   ├── mod /think → Explicație detaliată + clasificare
   └── mod /no_think → Doar clasificare rapidă
```

---

## 4.3 Funcții de Reward

Această secțiune explică **cum învață modelul** prin recompense și penalizări. Gândește-te la asta ca la dresarea unui câine: comportamentul bun primește recompensă, comportamentul greșit primește corecție.

---

### Privire de Ansamblu: Când Se Folosește Fiecare Reward

| Reward | Etapa 1 | Etapa 2 | Etapa 3 |
|--------|---------|---------|---------|
| Format | ✅ | - | ✅ |
| Soft Overlong | ✅ | - | ✅ |
| Accuracy | ✅ | - | ✅ |
| Hybrid Thinking | - | - | ✅ |
| Thinking Reward | - | - | ✅ |

---

### 1. Format Reward (r_fmt)

**Ce verifică:** A folosit modelul structura corectă de output?

**Formatul așteptat:**
```
<think>raționament aici...</think><answer>Real sau Fake</answer>
```

**Cum funcționează:**

| Situație | Reward | Explicație |
|----------|--------|------------|
| Format corect | r_fmt = 0 | Fără penalizare, modelul a urmat instrucțiunile |
| Format greșit | r_fmt = -1 | Penalizare pentru nerespectarea structurii |

**Exemplu:**
```
✅ BINE: <think>Lumina pare nenaturală...</think><answer>Fake</answer>
❌ RĂU:  Cred că e fals pentru că lumina arată ciudat.
```

**De ce contează:** Formatul consistent permite parsare automată și asigură că modelul oferă întotdeauna raționament înainte de răspuns.

---

### 2. Soft Overlong Reward (r_overlong)

**Ce verifică:** Este răspunsul prea lung?

**Problema:** Răspunsurile foarte lungi irosesc resurse de calcul și pot conține divagații. Dar tăierea bruscă e și ea problematică.

**Cum funcționează:** Folosește o penalizare "soft" cu o zonă tampon.

```
|←————————— Lmax (lungime maximă) ——————————→|
|←—— Zonă sigură ——→|←— Buffer (Lcache) —→|← Zonă penalizare →|
      r = 0              r = gradual            r = -1
```

**Formula:**

| Condiție | Reward | Semnificație |
|----------|--------|--------------|
| L_gen ≤ L_max - L_cache | r_overlong = 0 | Răspunsul e suficient de scurt, fără penalizare |
| L_max - L_cache < L_gen ≤ L_max | r_overlong = ((L_max - L_cache) - L_gen) / L_cache | În zona buffer: penalizare graduală (între 0 și -1) |
| L_gen > L_max | r_overlong = -1 | Prea lung, penalizare completă |

**Intuiție:**
- Dacă ești mult sub limită → nicio problemă
- Dacă te apropii de limită → avertisment blând (penalizare parțială)
- Dacă depășești limita → penalizare completă

**De ce "soft"?** În loc de o tăietură bruscă (0 sau -1), penalizarea graduală învață modelul să rămână natural în limite rezonabile.

---

### 3. Accuracy Reward (r_acc)

**Ce verifică:** A clasificat modelul corect?

**Cum funcționează:**

| Situație | Reward |
|----------|--------|
| Clasificare corectă (prezis Real când e Real, sau Fake când e Fake) | r_acc = +1 |
| Clasificare greșită | r_acc = 0 |

**Notă:** Răspunsurile greșite primesc 0, nu -1. Asta e intenționat—modelul nu e pedepsit pentru că încearcă, doar nu e recompensat.

**Exemplu:**
```
Video-ul este de fapt: FAKE
Modelul prezice: Fake  → r_acc = +1 ✅
Modelul prezice: Real  → r_acc = 0  ❌
```

---

### 4. Hybrid Thinking Reward (r_hybrid)

**Ce verifică:** A respectat modelul instrucțiunea de mod de gândire?

**Cele două moduri:**
- `/think` → Modelul TREBUIE să ofere raționament detaliat
- `/no_think` → Modelul TREBUIE să sară peste raționament (`<think></think>` gol)

**Cum funcționează:**

| Situație | Reward |
|----------|--------|
| Modelul urmează modul corect | r_hybrid = 0 |
| Modelul gândește când i s-a spus `/no_think` | r_hybrid = -1 |
| Modelul sare peste gândire când i s-a spus `/think` | r_hybrid = -1 |

**Exemple:**
```
Instrucțiune: /think
✅ BINE: <think>Fața arată mișcări nenaturale...</think><answer>Fake</answer>
❌ RĂU:  <think></think><answer>Fake</answer>  → Penalizare! Trebuia să raționeze.

Instrucțiune: /no_think
✅ BINE: <think></think><answer>Fake</answer>
❌ RĂU:  <think>Lasă-mă să analizez...</think><answer>Fake</answer>  → Penalizare! Trebuia să fie rapid.
```

**De ce contează:** Permite flexibilitate—analiză detaliată când e nevoie, răspunsuri rapide când viteza contează.

---

### 5. Thinking Reward (r_think)

**Ce verifică:** Este raționamentul de fapt de calitate bună?

**Provocarea:** Un model poate produce text care arată ca raționament dar e de fapt nonsens. Cum evaluăm calitatea?

**Soluția:** Folosim un model extern (SophiaVL-R1-Thinking-Reward-Model-3B) să judece.

**Cum funcționează:**

| Mod | Formulă | Explicație |
|-----|---------|------------|
| `/no_think` | r_think = 0 | Nu există raționament de evaluat |
| `/think` | r_think = min(r_acc, M(y_res)) | Scor de calitate de la modelul extern, dar plafonat de acuratețe |

Unde:
- M = model extern de evaluare (SophiaVL)
- y_res = răspunsul modelului
- 0 ≤ r_think ≤ 1

**De ce min(r_acc, ...)?**
- Dacă clasificarea e GREȘITĂ (r_acc = 0), calitatea raționamentului nu contează → r_think = 0
- Dacă clasificarea e CORECTĂ (r_acc = 1), atunci calitatea raționamentului contează → r_think = M(y_res)

**Intuiție:** Raționament bun care duce la răspuns greșit = inutil. Recompensăm raționamentul bun doar când produce rezultate corecte.

**Exemplu:**
```
Video: De fapt FAKE

Răspuns A:
<think>Lumina e inconsistentă și umbrele nu se potrivesc...</think>
<answer>Fake</answer>
→ Corect + Raționament bun = r_think ≈ 0.9 ✅

Răspuns B:
<think>Lumina e inconsistentă și umbrele nu se potrivesc...</think>
<answer>Real</answer>
→ Răspuns greșit = r_think = 0 (raționament bun irosit) ❌

Răspuns C:
<think>Nu știu, poate fake?</think>
<answer>Fake</answer>
→ Corect dar raționament slab = r_think ≈ 0.3 ⚠️
```

---

### Formulele Complete de Reward

**Etapa 1 (Foundation RL):**
```
R_stage-1 = r_fmt + r_overlong + r_acc
```
Interval: -2 până la +1

**Etapa 3 (Advanced RL):**
```
R_stage-3 = r_fmt + r_overlong + r_acc + r_hybrid + r_think
```
Interval: -3 până la +2

---

### Sumar Vizual

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUXUL FUNCȚIILOR DE REWARD                  │
└─────────────────────────────────────────────────────────────────┘

Modelul generează răspuns
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ VERIFICARE      │     │ VERIFICARE      │     │ VERIFICARE      │
│ FORMAT          │     │ LUNGIME         │     │ ACURATEȚE       │
│                 │     │                 │     │                 │
│ Format corect?  │     │ În limite?      │     │ Răspuns corect? │
│ Da → 0          │     │ Da → 0          │     │ Da → +1         │
│ Nu → -1         │     │ Buffer → parțial│     │ Nu → 0          │
│                 │     │ Depășit → -1    │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │     TOTAL ETAPA 1       │
                    │  R = fmt + overlong +   │
                    │        accuracy         │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │          DOAR ETAPA 3               │
              ▼                                     ▼
    ┌─────────────────┐               ┌─────────────────┐
    │ VERIFICARE      │               │ CALITATE        │
    │ HYBRID          │               │ RAȚIONAMENT     │
    │                 │               │                 │
    │ A urmat modul?  │               │ SophiaVL judecă │
    │ Da → 0          │               │ Scor 0 la 1     │
    │ Nu → -1         │               │ (doar dacă      │
    │                 │               │ /think + corect)│
    └────────┬────────┘               └────────┬────────┘
             │                                 │
             └─────────────┬───────────────────┘
                           │
              ┌────────────┴────────────┐
              │     TOTAL ETAPA 3       │
              │  R = Etapa1 + hybrid +  │
              │       thinking          │
              └─────────────────────────┘
```

---

## 5. Experimente

### Configurație Experimentală

| Parametru | Valoare |
|-----------|---------|
| Model de Bază | Qwen2.5-VL-7B-Instruct |
| Model Extern de Reward | SophiaVL-R1-Thinking-Reward-Model-3B |
| Sampling Video | 16 cadre la 4 FPS |
| Metodă Fine-tuning | LoRA (rank=16, alpha=32) |
| Learning Rate | 1×10⁻⁵ |
| Precizie | bfloat16 |
| Metrică Primară | Acuratețe (ACC) per subcategorie |

---

## 5.1 Benchmark-uri pe o Singură Modalitate

Aceste benchmark-uri testează modelul pe UN singur tip de conținut (fie imagini, fie video, nu ambele).

### Rezultate So-Fake-Set

| Model | Acuratețe | Scor F1 |
|-------|-----------|---------|
| **BusterX++** | **93.9%** | **93.7%** |
| BusterX++ (/no_think) | 92.3% | 92.1% |
| SOTA Anterior (So-Fake-R1) | 93.2% | 92.9% |

**Interpretare:** BusterX++ depășește state-of-the-art anterior cu **+0.7%**. Modul /no_think pierde doar 1.6% acuratețe dar e mai rapid.

### Rezultate GenBuster-200K

| Model | ACC Set Test | ACC Out-of-Domain |
|-------|--------------|-------------------|
| **BusterX++** | **88.3%** | **92.4%** |
| BusterX++ (/no_think) | 87.5% | - |
| BusterX (anterior) | 85.5% | 84.8% |

**Interpretare:**
- **+2.8%** îmbunătățire față de BusterX pe setul de test
- **+7.6%** îmbunătățire pe date out-of-domain (generalizare mai bună!)

---

## 5.2 Performanță Cross-Modal (Benchmark GenBuster++)

Acesta e benchmark-ul NOU creat de autori (Secțiunea 3.2) care testează ATÂT imagini CÂT ȘI video împreună.

### Rezultate Principale (Tabel 4)

| Model | Img Real | Img Fals | Vid Real | Vid Fals | **Overall** |
|-------|----------|----------|----------|----------|-------------|
| **BusterX++** | 80.4% | 76.2% | 95.3% | 57.9% | **77.5%** |
| BusterX++ (/no_think) | 80.5% | 74.4% | 96.4% | 55.9% | 76.8% |
| BusterX (anterior) | - | - | - | - | 68.3% |

**Comparații cu Baseline-uri (MLLM-uri generale fără fine-tuning):**

| Model | Acuratețe Overall |
|-------|-------------------|
| Qwen2.5-VL-7B | 55.4% |
| InternVL3-8B | 55.5% |
| MiniCPM-o 2.6 | 53.3% |

### Observații Cheie

| Constatare | Valoare | Interpretare |
|------------|---------|--------------|
| Video-uri reale - cele mai ușoare | 95.3% | Modelul excelează la confirmarea video-urilor autentice |
| Video-uri false - cele mai grele | 57.9% | Detectarea video-urilor false e provocatoare |
| Scădere acuratețe /no_think | -0.7% | Pierdere minimă pentru inferență mai rapidă |
| vs MLLM-uri generale | +22% | Antrenarea specializată contează enorm |
| vs BusterX | +9.2% | RL multi-etape îmbunătățește semnificativ |

**De ce sunt video-urile false cele mai grele?**
- Generatoarele state-of-the-art (Sora, Kling) produc conținut foarte realist
- Benchmark-ul a păstrat specific doar cele mai convingătoare falsuri
- Video-ul are mai multe dimensiuni (temporale) care pot fi corecte/greșite

---

## 5.3 Analiză Cold-Start vs Non-Cold-Start

Asta dovedește DE CE abandonarea cold-start-ului (Secțiunea 4.1) a fost alegerea corectă.

### Tabel Comparativ (Tabel 5)

| Strategie | Img Real | Img Fals | Vid Real | Vid Fals | **Overall** |
|-----------|----------|----------|----------|----------|-------------|
| Doar cold-start | 72.4% | 64.7% | 80.5% | 51.9% | 67.4% |
| Cold-start + Etapa 3 | 81.0% | 65.9% | 91.4% | 53.2% | 72.9% |
| **Fără cold-start + Etapa 3** | 81.2% | 76.7% | 94.1% | 57.5% | **77.4%** |

### Efectul de Crossover

```
Acuratețe
    ^
    │                                          ★ 77.4% (Fără cold-start)
80% │                                    ╱
    │                              ╱────╱
    │                        ╱────╱
75% │                  ╱────╱
    │            ╱────╱─────────────────── 72.9% (Cold-start)
    │      ╱────╱
70% │ ────╱
    │╱ 71.7%  (Cold-start începe mai sus)
    │  69.4%  (Fără cold-start începe mai jos)
    └────────────────────────────────────────> Etapa Antrenare
         Etapa 1        Etapa 2        Etapa 3
```

**Insight Cheie:**
- Cold-start: 71.7% → 72.9% (doar +1.2% câștig)
- Fără cold-start: 69.4% → 77.4% (**+8.0% câștig**)

Fără cold-start începe mai slab dar termină MULT mai bine pentru că nu moștenește bias-uri din date CoT potențial de calitate slabă.

---

## 5.4 Studii de Ablație

### Ce Date de Antrenare Contează? (Tabel 6)

| Date Antrenare | Img Real | Img Fals | Vid Real | Vid Fals | Overall |
|----------------|----------|----------|----------|----------|---------|
| Doar imagini | 78.7% | 77.2% | 77.7% | 52.1% | 71.4% |
| Doar video-uri | 75.9% | 67.9% | 95.9% | 51.9% | 72.9% |
| **Ambele (cross-modal)** | 80.4% | 76.2% | 95.3% | 57.9% | **77.5%** |

**Interpretare:** Antrenarea pe AMBELE modalități dă **+4.6%** față de doar video-uri. Învățarea cross-modal ajută modelul să generalizeze mai bine.

### Ce Etape de Antrenare Contează? (Tabel 7)

| Configurație | Acuratețe Overall |
|--------------|-------------------|
| Doar Etapa 1 | 69.4% |
| Etapa 1 + Etapa 2 | 69.3% |
| Etapa 1 + Etapa 3 | 77.4% |
| **Toate trei etapele** | **77.5%** |

**Interpretare:**
- Etapa 2 singură nu adaugă aproape nimic (+0.0%)
- Etapa 3 e critică (+8.0%)
- Etapa 2 e necesară doar pentru a activa raționamentul hibrid din Etapa 3

---

## 5.5 Evaluare Robustețe

Conținutul real adesea are compresie, zgomot sau blur. Funcționează modelul în continuare?

### Tipuri de Perturbări Aplicate

| Perturbație | Setări |
|-------------|--------|
| Compresie JPEG | quality=70 |
| Zgomot Gaussian | σ=5 |
| Blur Gaussian | standard |
| Cascadă | Toate combinate (stil Real-ESRGAN) |

### Rezultate Sub Perturbări (Tabel 8)

| Condiție | Img Real | Img Fals | Vid Real | Vid Fals | **Overall** |
|----------|----------|----------|----------|----------|-------------|
| **Curat** | 80.4% | 76.2% | 95.3% | 57.9% | **77.5%** |
| Doar JPEG | 82.1% | 67.2% | 94.5% | 55.6% | 74.9% |
| Doar Zgomot | 76.4% | 66.7% | 95.1% | 49.2% | 71.9% |
| Doar Blur | 91.6% | 66.4% | 93.9% | 57.6% | 77.4% |
| Cascadă (toate) | 90.8% | 53.5% | 97.0% | 40.8% | 70.5% |

### Analiză Robustețe

| Perturbație | Scădere Acuratețe | Evaluare |
|-------------|-------------------|----------|
| JPEG | -2.6% | Robustețe bună |
| Zgomot | -5.6% | Impact moderat |
| Blur | -0.1% | Robustețe excelentă |
| Cascadă | -7.0% | Provocatoare dar acceptabilă |

**Observații Cheie:**
- **Blur-ul ajută detecția realului** (91.6% vs 80.4%) - blur-ul face conținutul real mai ușor de identificat
- **Imaginile false suferă cel mai mult** sub cascadă (53.5%) - compresia distruge artefactele subtile ale falsurilor
- **Video-urile reale se îmbunătățesc** sub cascadă (97.0%) - poate pentru că degradarea le face să pară mai "naturale"

---

## 5.6 Studiu de Caz

Paper-ul oferă exemple vizuale care arată:

1. **Raționament Stabil:** Modelul identifică consistent aceleași artefacte în conținut similar
2. **Atenție la Detalii Low-level:** Modelul observă detalii subtile (reflexii nenaturale, inconsistențe de lumină)
3. **Inferență Bazată pe Cunoștințe:** Modelul folosește cunoștințe despre lume (ex: "acest politician nu ar fi în acest context")

---

## Sumar: Numere Cheie de Reținut

| Metrică | Valoare | Context |
|---------|---------|---------|
| **Cea mai bună acuratețe overall** | 77.5% | Pe GenBuster++ (cross-modal) |
| **Acuratețe So-Fake-Set** | 93.9% | Benchmark single-modality |
| **Acuratețe GenBuster-200K** | 88.3% | Benchmark la scară mare |
| **vs MLLM-uri generale** | +22% | BusterX++ vs Qwen2.5-VL |
| **vs BusterX** | +9.2% | Îmbunătățire din RL multi-etape |
| **Cold-start vs fără cold-start** | +4.5% | Fără cold-start câștigă după Etapa 3 |
| **Scădere acuratețe /no_think** | -0.7% | Cost minimal pentru inferență rapidă |
| **Categoria cea mai grea** | 57.9% | Video-uri false |
| **Categoria cea mai ușoară** | 95.3% | Video-uri reale |
| **Cea mai slabă robustețe** | -7.0% | Sub perturbație cascadă |

---
