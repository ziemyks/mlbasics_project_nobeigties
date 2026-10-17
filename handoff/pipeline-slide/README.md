# Prasības: slaids #7 "AI pipeline" prezentācijā

**Kur:** https://rsu-ai-project.vercel.app/#7, slaids `<section class="slide" data-phase="4" data-title="AI pipeline">`, kurā šobrīd ir placeholder "Pipeline content to be added".
**Kas:** placeholder vietā statiska pipeline shēma ar **vienu iepriekš ierakstītu testa darījumu**. Tā parāda, kā maksājums iziet cauri sistēmai līdz lēmumam, un ved uz demo slaidu (#8).
**Tikai prezentācijai:** slaids nesazinās ar API un nekur nesūta datus. Visi skaitļi ir iestrādāti HTML.

Šajā mapē:

| Fails | Kam |
|---|---|
| `pipeline_example.json` | Vienīgais datu avots slaidam: ierakstīta `POST /explain` atbilde testa darījumam |
| `screenshots/01–06_*.png` | Atsauce no demo lapas `localhost:3000/pipeline` ar **tieši šo darījumu**. Tie parāda saturu, nevis dizainu, ko kopēt. |

---

## 1. Testa piemērs (kas jāparāda)

Tiešsaistes pirkums Best Buy veikalā naktī, kartes īpašniekam ir 78 gadi. Modelis to bloķē ar 99,85% varbūtību.

| Lauks (`raw`) | Vērtība |
|---|---|
| merchant · category | Best Buy · `shopping_net` (online shopping) |
| amount | $2,334.03 |
| time | 2026-10-17 23:29 (sestdiena) |
| cardholder | dzimis 1947-12-13 → 78 gadi, Seattle, WA |
| fraud probability | **99.85 %** → **FRAUD / Block** |

Dati ir sintētiski: koordinātas neatbilst pilsētai. **Slaidā nerādīt lat/long**, rādīt tikai pilsētu, štatu un attālumu.

## 2. Saturs: 6 soļi, katram IN → OUT

Teksts slaidam angļu valodā, jo visa prezentācija ir angliski. Skaitļi ņemti no `pipeline_example.json`.

| # | Solis (virsraksts slaidā) | IN (ko saņem) | OUT (ko nodod tālāk) | JSON lauks |
|---|---|---|---|---|
| 01 | **Payment arrives** | card payment, 14 fields | Best Buy · $2,334.03 · 23:29 · online · Seattle, WA | `raw` |
| 02 | **Check the request** | 14 fields | 10 / 10 required fields present ✓ | `validation` |
| 03 | **Feature engineering** | time, date of birth, home + shop location, amount | hour **23** · age **78** · distance **1,757 km** · log-amount **7.76** | `engineered` |
| 04 | **Encode categories** | category `shopping_net`, state `WA` | 27 yes/no columns, 2 of them = 1 → **39 inputs** | `encoding` |
| 05 | **LightGBM model** | 39 numbers | 300 trees add up a score (**+6.50**), the sigmoid turns it into **99.85 %** | `raw_score`, `probability`, `boosting` |
| 06 | **Decision** | 99.85 % | **Block → analyst review** | `verdict`, `is_fraud` |

**"Why" (obligāti, tas ir slaida galvenais vēstījums):** kuras pazīmes virzīja lēmumu (`contributions`, SHAP log-odds vienībās). Rādīt top 4–5:

| Pazīme | Ieguldījums | Virziens |
|---|---|---|
| Amount (log) | +10.44 | → fraud |
| Hour (23:00) | +3.18 | → fraud |
| Age (78) | +1.97 | → fraud |
| Distance (1,757 km) | −0.74 | → legit |
| Category (online shopping) | +0.37 | → fraud |

Distance virza uz **leģitīmu**, un tas apstiprina 5. slaida atziņu "the assumption that failed". Vērts to pateikt skaļi.

**Scaling solis apzināti nav iekļauts.** Demo API vairs nemērogo datus, jo mērogošana ir iebūvēta modeļa sliekšņos un prognozes paliek identiskas (skat. `fold_scaler_into_model.py`).

## 3. Izkārtojums un tehniskie ierobežojumi

Prezentācija ir viens statisks HTML fails. Slaidi ir fiksētā **1280×720 px** izmērā un tiek mērogoti ekrānam. Satura laukums ir ~**1136 × 480 px** zem `kicker` + `h2`.

```
┌ kicker: BUILD · PIPELINE ─────────────────────────────────────────────────┐
│ h2: From payment to verdict in six steps                                  │
│                                                                           │
│ [01 Payment]→[02 Check]→[03 Features]→[04 Encode]→[05 LightGBM]→◆06 Block │  ~150 px
│  OUT value    OUT value    OUT value    OUT value    99.85 %              │
│                                                                           │
│ ┌ Why this payment was blocked ───────────────┐ ┌ Verdict ─────────────┐  │  ~220 px
│ │ Amount     ████████████████ +10.44           │ │ FRAUD · 99.85 %      │  │
│ │ Hour 23    █████ +3.18                       │ │ Block → analyst      │  │
│ │ Age 78     ███ +1.97                         │ │ review               │  │
│ │ Distance   ▌−0.74                            │ └──────────────────────┘  │
│ └──────────────────────────────────────────────┘                          │
│ (optional) after the verdict: analyst label → nightly retrain → …         │  ~40 px
└───────────────────────────────────────────────────────────────────────────┘
```

Noteikumi:

1. **Bez API un bez fetch.** Vajadzīgos laukus no `pipeline_example.json` ievietot kā JS konstanti vai tieši HTML, lai slaids strādātu arī PDF eksportā un bez interneta.
2. **Bez jaunām bibliotēkām.** Lietot HTML/CSS un inline SVG. Chart.js jau ir ielādēts, bet "Why" stabiņiem pietiek ar HTML div.
3. **Prezentācijas stils, nevis demo (Indexo) stils.** Lietot esošos CSS mainīgos: `--burgundy` struktūrai un aktīvajam solim, `--fraud` (#9b2335) "towards fraud" un Block, `--good` (#23679c) "towards legit", `--card` / `--line` kartītēm, `--serif` virsrakstiem, `--mono` vērtībām. Krāsu pāris `--fraud` / `--good` ir pārbaudīts (krāsu akluma separācija ΔE 16,6, kontrasts ≥ 3:1). Virzienu tik un tā rāda arī stabiņa puse un skaitlis ar zīmi.
4. **Nekonfliktēt ar prezentācijas kodu.** Visam CSS lietot prefiksu (piem. `.pl-…`) vai ierobežot to ar `section[data-title="AI pipeline"]`. JS ievietot IIFE. Neizmantot globālos nosaukumus `slides`, `show`, `current`, `PHASES`, `fit`.
5. **Netvert prezentācijas taustiņus:** ← → Space PageUp PageDown Home End N F.
6. **Statiskais stāvoklis ir galvenais.** Bez JS un drukā (`@media print`, 1 slaids = 1 lapa) jāredz pilna, aizpildīta shēma: visi OUT, "Why" un verdikts.
7. **Animācija ir neobligāta.** Ja to veido:
   - kad slaids kļūst `.active` (MutationObserver uz `class`), marķieris iziet soļus 01→06 aptuveni 6 s laikā, un pēc tam paliek galīgais stāvoklis;
   - klikšķis uz soļa izceļ tā IN/OUT;
   - neliela poga "Replay";
   - ja ieslēgts `prefers-reduced-motion`, animāciju nerādīt.
8. Skaitļus formatēt kā pārējā prezentācijā: `$2,334.03`, `99.85 %`, `1,757 km`.

## 4. Jāizlemj pirms darba (komandas lēmums)

1. **Viens slieksnis vai trīs zonas?** 6. slaidā ir "approve < 0.1 · analyst 0.1–0.4 · block ≥ 0.4", bet demo API izmanto vienu slieksni **0.5**. *Ieteikums:* 06. solī rādīt trīs zonas, lai stāsts sakristu ar 6. slaidu, un mazā fontā piebilst "live demo uses a single 0.5 cut-off". Ar p = 0.9985 abos variantos iznākums ir Block, tāpēc piemērs paliek pareizs.
2. **Vai rādīt feedback loop?** Demo lapā ir simulēts cikls: analītiķa iezīme / klienta sūdzība → iezīmētie dati → OpenClaw nakts pārtrenēšana → salīdzinājums ar etalonu → datu zinātnieka apstiprinājums → publicēšana (`06_full_schematic_with_feedback_loop.png`). *Ieteikums:* uz #7 to nerādīt vai rādīt tikai vienā rindā ar uzrakstu "planned". Tas pārklājas ar 9. slaidu ("keep it learning"), un uz #7 nepietiks vietas.
3. **Vai atjaunot 8. slaida (demo) piezīmes?** Tur rakstīts "The trained LightGBM model is saved together with the scaler…". Tagad mērogošana ir iebūvēta modelī. Ieteicamais teksts: *"The trained LightGBM model is saved with the exact list and order of its 39 input columns; the scaling step is folded into the model's thresholds, so the API feeds it raw values."*

## 5. Runātāja piezīmju melnraksts (`<aside class="notes">`)

```html
<div class="say">
<ul>
  <li>(~45 s) "Let's follow one real test payment through the system: an online purchase at Best Buy, $2,334, at 23:29, by a 78-year-old cardholder."</li>
  <li>The API first checks the request, then turns raw fields into the signals the model learned: hour 23, age 78, distance, log-amount. Categories become 27 yes/no columns, 39 numbers in total.</li>
  <li>LightGBM's 300 small trees add up a score; the sigmoid turns it into a probability: 99.85% fraud, so the payment is blocked and goes to an analyst.</li>
  <li>Why? Mostly the amount, the late hour and the age. Distance actually pushed towards legitimate, the same lesson as on the data slide.</li>
  <li>"Let's see it in action."</li>
</ul>
</div>
<div class="more">
  <p><strong>What is shown.</strong> One recorded response of the demo API (POST /explain) for a single synthetic test payment; the slide does not call the API.</p>
  <p><strong>How the score is built.</strong> Each of the 300 trees adds a small correction to a starting value (−9.23 in log-odds); the total for this payment is +6.50, and the sigmoid 1/(1+e^−score) gives 0.9985. The "why" bars are SHAP contributions computed by LightGBM itself (pred_contrib), and they add up exactly to the score.</p>
  <p><strong>No scaling step.</strong> The model was trained on standardised inputs, but trees only compare a value with a threshold, so the scaler was folded into the thresholds: identical predictions, one step fewer.</p>
  <p><strong>Lecture 6 mapping.</strong> This slide is the "serve" step (train → export → catalogue → build → deploy → serve → watch); "watch" and retraining are covered on the launch slide.</p>
</div>
```

## 6. TODO

**Sagatavošana**
- [ ] Izlemt §4.1 (viens slieksnis vai trīs zonas) un §4.2 (feedback loop jā/nē)
- [ ] Iegūt prezentācijas HTML avotu (Vercel projekts `rsu-ai-project`, nav šajā repo) un atrast sadaļu `7 · AI PIPELINE (PLACEHOLDER)`

**Izstrāde**
- [ ] Aizstāt `.placeholder` bloku. Atstāt `kicker` "Build · Pipeline", nomainīt `h2` (piem. "From payment to verdict in six steps")
- [ ] 6 soļu rinda ar bultām. Katrā solī numurs, virsraksts un OUT vērtība no §2
- [ ] 06. solis kā lēmuma rombs ar verdiktu Block un 99.85 %
- [ ] "Why" bloks: top 4–5 ieguldījumi kā diverģējoši stabiņi no nulles ass ar skaitli un zīmi (→ fraud `--fraud`, → legit `--good`)
- [ ] Verdikta kartīte: FRAUD · 99.85 % · "Block → analyst review"
- [ ] Datus iestrādāt inline no `pipeline_example.json` (bez fetch)
- [ ] CSS ar prefiksu, JS IIFE, bez globāliem nosaukumiem un bez taustiņu tveršanas
- [ ] (neobligāti) Animācija, klikšķis uz soļa, Replay, `prefers-reduced-motion`
- [ ] Aizstāt `<aside class="notes">` ar §5 tekstu
- [ ] (ja izlemts) Atjaunot 8. slaida piezīmes pēc §4.3

**Pārbaude**
- [ ] `#7` atveras tieši šis slaids. Navigācija ← → un N/F strādā kā iepriekš
- [ ] Viss satilpst 1280×720 bez pārplūšanas: pārbaudīt 1440×900 un 1920×1080 ekrānā un projektorā
- [ ] PDF eksports (pārlūka Print, 1280×720): slaids redzams pilnā, aizpildītā stāvoklī
- [ ] Bez interneta: slaids pilnībā strādā, tikai Chart.js CDN var neielādēties citos slaidos
- [ ] Visi skaitļi sakrīt ar `pipeline_example.json` (99.85 %, +6.50, $2,334.03, 1,757 km, 78, 23)
- [ ] Teksts lasāms no zāles: vērtībām ≥ 13 px, virsrakstiem ≥ 18 px
- [ ] Pārejas teikums uz demo: "Let's see it in action." Demo (#8) atrāda to pašu plūsmu dzīvajā lapā `localhost:3000/pipeline`

## 7. Ja piemēru vajag ierakstīt no jauna

Ja mainās modelis vai API, palaid demo (`./start_demo.sh`) un ieraksti to pašu darījumu vēlreiz:

```bash
jq '.raw' handoff/pipeline-slide/pipeline_example.json > /tmp/raw.json
curl -s -X POST http://127.0.0.1:5050/explain \
     -H "Content-Type: application/json" -d @/tmp/raw.json | jq .
```

Pēc tam atjaunini skaitļus §2 un slaidā. Ar pašreizējo modeli rezultātam jābūt `probability: 0.998493`, `raw_score: 6.4959`.
