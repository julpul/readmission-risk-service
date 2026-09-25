# Krok 5 — trening, progi i metryki

Stan na 2026-09-02. Liczby pochodzą z `uv run python scripts/trace_baseline.py` (`N_SPLITS=3`)
oraz ze skryptów weryfikacyjnych uruchomionych na tej samej kohorcie. Wszystkie wartości
w tym dokumencie są policzone, nie oszacowane.

## Oś całości: co wywołuje co

`run_experiment` jest jedyną funkcją, która zna pełną historię. Reszta to klocki.

```
data = (X_train, X_test, y_train, y_test)      <- z patient_level_split
   │
   ├─ 1. cross_val_predict(pipeline, X_train, y_train, cv, groups)
   │        trenuje pipeline n_splits razy
   │        zwraca oof_proba: każdy wiersz treningu oceniony przez model,
   │        który tego wiersza NIE widział
   │
   ├─ 2. find_threshold(y_train, oof_proba, criterion)
   │        wybiera JEDEN próg na podstawie danych treningowych
   │
   ├─ 3. train_model(...)  -> pipeline.fit(X_train, y_train)
   │        finalny model, tym razem na 100% treningu
   │
   ├─ 4. pipeline.predict_proba(X_test)[:, 1]
   │        pierwszy i jedyny kontakt z testem
   │
   └─ 5. compute_metrics(y_test, test_proba, threshold)
```

Kolejność nie jest przypadkowa i to jest najważniejsza rzecz w tym pliku. Próg wybierasz
w kroku 2, a test dotykasz dopiero w kroku 4. Gdybyś wybrał próg na teście, raportowana
precyzja byłaby najlepszą z kilkudziesięciu tysięcy możliwości — czyli optymistycznym
kłamstwem. Rozdzielenie „na czym uczę", „na czym wybieram próg" i „na czym raportuję"
to jest cały mechanizm uczciwości tego kodu.

---

## `evaluate.py` linijka po linijce

Nagłówek modułu mówi „pure functions: arrays in, numbers out" i to nie jest kosmetyka.
Ten plik nie importuje niczego z `readmission` — nie wie o pipeline'ach ani o pandas.
Dzięki temu przetestujesz go tablicą `[0.1, 0.9, 0.4]` napisaną ręcznie, bez ładowania
101 766 wierszy CSV.

### `compute_metrics`

```python
y_pred = (y_proba >= threshold).astype(int)
```

`>=`, nie `>`. To nie jest wybór stylistyczny. `precision_recall_curve` w sklearn używa
wewnętrznie `y_score >= threshold`, więc gdybyś tu dał `>`, metryki policzone dla progu
wziętego z tej krzywej **nie zgadzałyby się z tą krzywą** dokładnie w punkcie, który
wybrałeś. Różnica dotyczy wierszy o prawdopodobieństwie równym progowi co do bitu — a próg
*pochodzi* z wartości prawdopodobieństwa jednego z wierszy, więc taki wiersz zawsze istnieje.

`.astype(int)` zamienia maskę boolowską na 0/1, bo `precision_score` i spółka oczekują
etykiet, nie masek.

```python
"threshold": float(threshold),
"pr_auc": float(average_precision_score(y_true, y_proba)),
```

`float(...)` wokół każdej wartości: sklearn zwraca `np.float64`. Dopóki trzymasz to w pamięci,
różnicy nie ma, ale w momencie zapisu do JSON dostaniesz `TypeError: Object of type float64
is not JSON serializable`, a w CSV — brzydkie `np.float64(0.1387)`. Ten `float()` jest po to,
żeby wynik eksperymentu dał się zapisać na dysk bez dodatkowej obróbki. To samo `int()` przy
`n_flagged`.

Trzy pierwsze metryki nie zależą od progu, cztery kolejne zależą. W outpucie widać to wprost —
`pr_auc`, `roc_auc` i `brier` są identyczne dla obu kryteriów, a `precision` skacze z 0.1244
na 0.1977. To jest podział na dwa różne pytania:

- **Czy model umie uszeregować pacjentów od najbardziej do najmniej ryzykownego?**
  → `pr_auc`, `roc_auc`. Odporne na próg, bo patrzą na wszystkie progi naraz.
- **Czy ta konkretna decyzja operacyjna ma sens?**
  → `precision`, `recall`, `f1`, `n_flagged`. Zależne od progu, bo próg *jest* decyzją.

`average_precision_score`, a nie `auc(recall, precision)` — bo AP sumuje prostokąty
(`Σ (R_n − R_{n−1}) · P_n`), a `auc` interpoluje trapezami. Krzywa PR przy rzadkiej klasie
pozytywnej ma pionowe skoki i trapez zawyża pole pod nimi. AP jest tu jedyną poprawną wersją.

`zero_division=0` — jeśli próg jest tak wysoki, że nikt nie zostaje oflagowany, precyzja
to `0/0`. Bez tego parametru dostajesz ostrzeżenie i `nan`, które zatruje całą tabelę
eksperymentów. Ale uwaga na pułapkę interpretacyjną: `precision=0.0` znaczy wtedy „model
nic nie zgłosił", a nie „model się mylił". Rozróżnisz te dwa przypadki **tylko** dzięki
`n_flagged` — i to jest prawdziwy powód, dla którego ta pozycja jest w słowniku.

`n_flagged` to zresztą jedyna metryka, którą rozumie ordynator. 5 256 flag z 13 998 pacjentów
znaczy „przydzielcie koordynatora do 38% wypisów". Żaden szpital tego nie zrobi.

### `find_threshold`

```python
precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
precision, recall = precision[:-1], recall[:-1]
```

To jest miejsce, gdzie łatwo o cichy błąd, więc rozłóżmy je na czynniki. Zweryfikowane
na treningu:

| tablica | długość |
|---|---|
| `precision`, `recall` | 55 948 |
| `thresholds` | 55 947 |
| wierszy w `X_train` | 55 992 |
| unikalnych wartości `oof` | 55 947 |

Dwie rzeczy się tu dzieją naraz.

Po pierwsze, punktów jest **mniej niż wierszy** (55 947 < 55 992). Krzywa ma jeden punkt
na każdą *unikalną* wartość prawdopodobieństwa, a 45 wierszy dostało prawdopodobieństwo
identyczne z jakimś innym wierszem. Nie da się ich rozdzielić żadnym progiem, więc dzielą
punkt na krzywej.

Po drugie, `precision` i `recall` mają o jeden element więcej niż `thresholds`. sklearn
docina na końcu sztuczny punkt `(precision=1.0, recall=0.0)` — „gdybyś nie oflagował nikogo,
z definicji nie masz fałszywych alarmów". Za tym punktem **nie stoi żaden próg**, dlatego
`thresholds` jest krótsze. Stąd `[:-1]`: bez tego `thresholds[i]` dla `i = 55947` rzuci
`IndexError`, a jeśli akurat nie rzuci, to zwróci próg odpowiadający zupełnie innemu punktowi
krzywej. To jest klasyczny błąd, który nie wywala programu, tylko po cichu zwraca zły wynik.

```python
f1 = 2 * precision * recall / (precision + recall + EPS)
return float(thresholds[f1.argmax()])
```

`EPS = 1e-12` w mianowniku zamiast `np.errstate` czy maskowania: gdy precyzja i czułość
są jednocześnie zerowe, dostajesz `0/1e-12 = 0` zamiast `nan` plus ostrzeżenie. Tanio i czytelnie.

`argmax` zwraca **pierwszy** indeks maksimum. Przy remisie F1 wybierze więc niższy próg,
czyli wariant o wyższej czułości. To sensowna domyślna preferencja w problemie klinicznym,
ale jest w kodzie milcząco — warto wiedzieć, że tam jest.

```python
mask = precision >= min_precision
candidates = np.flatnonzero(mask)
best = candidates[recall[mask].argmax()]
```

Ten trzykrok wygląda na okrężny, ale jest dokładnie poprawny i warto rozumieć dlaczego.
`recall[mask]` to **skrócona** tablica — jeśli maska ma 2 084 elementów, to `argmax` zwraca
liczbę z zakresu 0–2083. Ta liczba nie jest indeksem w `thresholds`, która ma 55 947 pozycji.
`candidates` to mapa z powrotem: „element numer *k* skróconej tablicy siedział pierwotnie
pod indeksem `candidates[k]`". Najczęstszy błąd w tym miejscu to `thresholds[recall[mask].argmax()]` —
kod wykonuje się bez błędu i zwraca próg z zupełnie innej części krzywej.

Dlaczego argmax po czułości, a nie po prostu „najniższy próg spełniający warunek"? Bo krzywa
PR **nie jest monotoniczna**. Sprawdzone: powyżej wybranego progu leży 2 090 punktów, ale tylko
2 084 z nich ma precyzję ≥ 0.20 — sześć punktów wpada z powrotem poniżej progu. Wersja
z `argmax` po czułości nie zakłada monotoniczności i dlatego jest odporna.

Komunikat błędu podaje `best achievable` zamiast samego „nie da się". Różnica praktyczna:
gdy poprosisz o precyzję 0.60, dowiesz się, że sufit to 0.50, więc wiesz, czy negocjować
z biznesem o 0.05 czy przebudować cechy.

---

## `train.py` linijka po linijce

```python
MODELS = {"logreg": LogisticRegression}
```

W słowniku siedzą **klasy, nie instancje**. To celowe. Gdyby były instancje, wszystkie
eksperymenty współdzieliłyby jeden obiekt — a estymator sklearn po `fit` przechowuje stan
(`coef_`, `n_iter_`), więc drugi eksperyment nadpisywałby pierwszy. Do tego nie dałoby się
wstrzyknąć `params`. Klasa plus `(**params)` daje świeży, niezależny model na każde wywołanie.

Praktyczny zysk: `run_experiment("logreg", ...)` sterowane jest **stringiem**, więc konfiguracja
eksperymentu może przyjść z pliku YAML albo z argumentu CLI, bez importowania sklearn.

```python
BASELINE_PARAMS = {"class_weight": "balanced", "max_iter": 1000, "random_state": 42}
```

`max_iter=1000` nie jest zabobonem — trace pokazuje `n_iter_ = 146`. Domyślne `max_iter=100`
w sklearn **nie wystarczyłoby**, dostałbyś `ConvergenceWarning` i niedouczone współczynniki.
To jest dokładnie ten rodzaj liczby, który warto mieć potwierdzony, a nie przepisany z tutoriala.

`class_weight="balanced"` jest problematyczne i wracam do tego w sekcji z odkryciami.

```python
def train_model(model_name, params, X_train, y_train) -> Pipeline:
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name!r}; available: {sorted(MODELS)}")
```

Walidacja przed pracą. `{model_name!r}` daje cudzysłowy w komunikacie, więc odróżnisz
`'logreg '` ze spacją od `'logreg'`. `sorted(MODELS)` pokazuje, co jest dostępne — komunikat
błędu, który sam siebie tłumaczy.

```python
    return pipeline.fit(X_train, y_train)
```

`fit` w sklearn zwraca `self`, więc to jedno wyrażenie robi trening i zwrot. Idiom, nie skrót.

```python
def train_baseline(X_train, y_train) -> Pipeline:
    return train_model("logreg", BASELINE_PARAMS, X_train, y_train)
```

Jedno miejsce prawdy o tym, czym jest „baseline". Gdy za miesiąc porównasz gradient boosting
do baseline'u, nie będziesz zgadywał, jakie parametry miał tamten przebieg.

### `run_experiment` — sedno

```python
oof_proba = cross_val_predict(..., method="predict_proba")[:, 1]
```

`cross_val_predict` robi coś innego niż `cross_val_score`. Nie zwraca metryki — zwraca
**predykcje**, poskładane z powrotem w oryginalnej kolejności wierszy. Każdy wiersz dostaje
prognozę z tego foldu, w którym był w części walidacyjnej, czyli od modelu, który go nie widział.
Dlatego dopiero na tych liczbach wolno wybierać próg.

`method="predict_proba"` daje macierz `(n, 2)`: kolumna 0 to P(brak readmisji), kolumna 1
to P(readmisja). Sumują się do 1 (trace to pokazuje: `0.4906 + 0.5094 = 1.0`). `[:, 1]`
wybiera klasę pozytywną. Kolejność kolumn odpowiada `clf.classes_`, czyli posortowanym
etykietom — dla `{0, 1}` klasa 1 jest druga. To działa, ale opiera się na konwencji; przy
etykietach tekstowych warto by było użyć `classes_` jawnie.

```text
groups=X_train["patient_nbr"],
```

**Uwaga — to dziś nic nie robi.** Sprawdzone: w kohorcie jest 69 990 wierszy i 69 990
unikalnych `patient_nbr`, maksymalna liczba wizyt na pacjenta wynosi 1. `build_cohort`
wywołuje `duplicate_patients(out, "first")`, więc każdy pacjent występuje raz i grupy są
jednoelementowe. `StratifiedGroupKFold` degeneruje się wtedy do zwykłego `StratifiedKFold`,
a „shared patients: 0" w trace'ie jest prawdą trywialną, nie testem.

To **nie znaczy, że kod jest zły** — znaczy, że jest przygotowany na `strategy="all"`.
Ale nie wolno o nim myśleć jak o działającym dziś zabezpieczeniu, bo gdy zmienisz strategię
i coś zepsujesz, żaden dzisiejszy test tego nie złapie. Ta sama uwaga dotyczy
`GroupShuffleSplit` w `patient_level_split`.

```python
pipeline = train_model(model_name, params, X_train, y_train)
```

Model finalny uczy się na **100%** treningu, a próg pochodził od modeli uczonych na 2/3
(przy `n_splits=3`). To są różne modele. Więcej danych → pewniejsze współczynniki → rozkład
prawdopodobieństw lekko się rozjeżdża względem tego, na którym wybierałeś próg. Przeniesienie
progu jest więc przybliżeniem — i w sekcji odkryć widać, ile to kosztuje.

```python
return {"model": model_name, "criterion": criterion, **params, **metrics}
```

Spłaszczenie do jednego wiersza tabeli eksperymentów — dokładnie to, czego chcesz do
`pd.DataFrame(results)`. Dwie miny: gdyby `params` zawierało klucz kolidujący z metryką
(np. `threshold`), `**metrics` po cichu by go nadpisało; a `class_weight` podany jako słownik
zapisze się do CSV jako nieczytelny string.

---

## Output `trace_baseline.py`, krok po kroku

**STEP 1–2.** 101 766 → 69 990 wierszy, odpada 31 776. Target rate 8.98%. Te liczby zgadzają
się z `docs/features_notes.md`, więc kohorta się nie zmieniła.

**STEP 3.** 55 992 / 13 998 = 80/20. `X_train` ma 50 kolumn, kohorta miała 51 — zniknął `target`,
i to sprawdza linijka `'target' still in X_train? False`. Warto ją rozumieć jako asercję,
nie ozdobnik: gdyby `target` przeszedł dalej, `ColumnTransformer` i tak by go wyciął
(`remainder="drop"`), ale przy pierwszej zmianie list kolumn dostałbyś model o AUC 1.00.

**STEP 4.** Pięć kroków pipeline'u, nic nie jest jeszcze wytrenowane. `Pipeline` to przepis.
Ta różnica między „obiektem przepisu" a „obiektem wytrenowanym" jest w sklearn źródłem
większości nieporozumień — ten sam obiekt pełni obie role, a rozróżnia je obecność atrybutów
z podkreślnikiem na końcu (`coef_`, `n_iter_`, `classes_`).

**STEP 5 — tu jest pierwszy sygnał ostrzegawczy.**

```
min=0.027  max=0.999  mean=0.480
actual positive rate: 0.090
```

Średnie przewidziane prawdopodobieństwo to 0.48, a faktyczny odsetek readmisji to 0.09.
Model twierdzi, że co drugi pacjent wróci. To **nie jest błąd w kodzie** — to bezpośredni
skutek `class_weight="balanced"`, który mnoży wagę klasy pozytywnej ok. 10×, żeby obie klasy
ważyły w funkcji straty tyle samo. Model uczy się w świecie, w którym readmisja jest zjawiskiem
50-procentowym. Konsekwencje w sekcji poniżej.

**STEP 6.** Tabela punktów pracy to jest cała prawda o tym modelu w pięciu wierszach:

| próg | precyzja | czułość |
|---:|---:|---:|
| 0.3235 | 0.0923 | 0.9777 |
| 0.4138 | 0.1014 | 0.8477 |
| 0.4765 | 0.1150 | 0.6407 |
| 0.5405 | 0.1329 | 0.3701 |
| 0.6527 | 0.1815 | 0.1011 |

Żeby podnieść precyzję z 9% (czyli z poziomu losowego zgadywania) do 18%, oddajesz 90%
przypadków. Tego nie naprawi żaden próg — to jest sufit rankingu, a ranking naprawia się
tylko lepszymi cechami albo lepszym modelem.

**STEP 7.** `f1 → 0.5052`, `recall_at_precision → 0.6743`. Próg dla F1 wypada tuż obok 0.5,
ale to zbieg okoliczności wynikający z `balanced`, a nie żadna głęboka prawda o 0.5.

**STEP 8.** `coef_` ma kształt `(1, 119)` — **jeden** wiersz, nie dwa. Regresja logistyczna
binarna modeluje jedną granicę decyzyjną; współczynniki dla klasy 0 to dokładnie minus
współczynniki klasy 1, więc sklearn trzyma tylko jeden komplet. 119 współczynników = 119 cech
po preprocessingu (`features_notes.md` notował 113 — przybyło 6, bo doszły kolumny do list).

**STEP 9–10.** Test dotknięty raz. I tu najciekawsze:

```
pr_auc  0.1387     positive rate 0.0902   ->  1.54x lepiej niż losowo
roc_auc 0.6068
brier   0.2379
```

---

## Cztery rzeczy, których output nie mówi wprost

### 1. `class_weight="balanced"` nie daje nic, a psuje Briera

Porównanie na tym samym splicie, oba modele wytrenowane i ocenione tak samo:

| wersja | pr_auc | roc_auc | brier | średnie p |
|---|---:|---:|---:|---:|
| `class_weight="balanced"` | 0.1387 | 0.6068 | **0.2379** | 0.480 |
| bez `class_weight` | 0.1385 | 0.6058 | **0.0809** | 0.090 |

Korelacja rang między predykcjami obu wersji: **0.9941**. To jest praktycznie ten sam ranking
pacjentów. Jakość uszeregowania nie zmienia się w trzecim miejscu po przecinku, a Brier
poprawia się trzykrotnie.

Skala Briera: stały predyktor mówiący każdemu „9.02%" osiąga **0.0820**. Twój obecny model
ma 0.2379, czyli jest **2.9× gorszy od predyktora, który w ogóle nie patrzy na pacjenta**.
Wersja bez `class_weight` ma 0.0809 — nieznacznie lepiej od stałej, czyli jej prawdopodobieństwa
faktycznie coś znaczą.

Widać to też w decylach modelu `balanced` (test):

| decyl | średnie przewidziane p | faktyczny odsetek |
|---:|---:|---:|
| 0 (najniższy) | 0.308 | 0.041 |
| 5 | 0.488 | 0.087 |
| 9 (najwyższy) | 0.674 | 0.163 |

Ranking działa (0.041 → 0.163, prawie 4× lift), ale liczby są zawyżone 4–7×. Jeśli aplikacja
ma kiedykolwiek pokazać lekarzowi „ryzyko 51%", to jest to liczba nieprawdziwa.

**Sedno:** `class_weight="balanced"` i strojenie progu rozwiązują **ten sam problem** —
niezbalansowane klasy — tylko że pierwsze robi to przez zniekształcenie prawdopodobieństw,
a drugie przez przesunięcie punktu decyzyjnego. Robisz obie rzeczy naraz. Skoro i tak
wybierasz próg jawnie, `class_weight` jest zbędny i kosztuje kalibrację.

### 2. Kontrakt „precyzja ≥ 0.20" nie został dowieziony

```
OOF:  próg 0.6743, precyzja 0.2000
TEST: próg 0.6743, precyzja 0.1977
```

Poprosiłeś o 0.20, dostałeś 0.1977. To nie jest szum ani pech — to systematyczna właściwość
tej procedury. `find_threshold` wybiera punkt, który **dokładnie** dotyka granicy (418 TP
na 2 090 flag = idealne 0.200000). Punkt leżący dokładnie na granicy ma z definicji około
50% szans, że na nowych danych wypadnie poniżej.

Policzony błąd standardowy precyzji w tym punkcie: przy 2 090 oflagowanych to ±0.0087,
czyli 95% przedział ufności **[0.183, 0.217]**. Granica 0.20 leży w samym środku tego przedziału.

Lekarstwo — margines. Sprawdzone empirycznie:

| cel na OOF | próg | precyzja OOF | **precyzja TEST** | czułość TEST | oflagowanych |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.6743 | 0.2000 | **0.1977** ✗ | 0.0800 | 511 |
| 0.22 | 0.6902 | 0.2201 | **0.2039** ✓ | 0.0666 | 412 |
| 0.24 | 0.7086 | 0.2400 | **0.2273** ✓ | 0.0555 | 308 |
| 0.26 | 0.7350 | 0.2601 | **0.2673** ✓ | 0.0460 | 217 |

Celowanie w 0.22 na OOF dowozi 0.20 na teście, kosztem spadku czułości z 0.080 na 0.067.
To jest świadomy zakup zapasu, nie strojenie pod wynik.

### 3. Wąskim gardłem nie są dane, tylko cechy

`features_notes.md` zapisało obawę, że reguła „pierwsza wizyta na pacjenta" wycina sygnał.
Obawa jest uzasadniona i mierzalna:

| | wszystkie wizyty | pierwsze wizyty |
|---|---:|---:|
| średnie `number_inpatient` | 0.631 | 0.176 |
| odsetek z `number_inpatient` > 0 | 33.3% | 11.7% |
| target rate | 11.4% | 9.0% |

Reguła „pierwsza wizyta" usuwa dwie trzecie pacjentów z historią hospitalizacji — czyli
dokładnie tę cechę, która w literaturze jest najsilniejszym predyktorem.

**Ale eksperyment pokazał coś przeciwnego niż oczekiwałem.** Wytrenowałem ten sam pipeline
na wszystkich wizytach pacjentów treningowych (79 541 wierszy zamiast 55 992) i oceniłem
na **identycznym** zbiorze testowym pierwszych wizyt, bez ani jednego wspólnego pacjenta:

| trening | pr_auc | roc_auc |
|---|---:|---:|
| pierwsze wizyty (55 992) | 0.1387 | 0.6068 |
| wszystkie wizyty (79 541) | 0.1381 | 0.6038 |

Zero poprawy, wręcz cień pogorszenia. Wniosek jest mocniejszy niż samo „nie warto":
**dodanie 42% danych treningowych nie ruszyło wyniku ani o włos, więc model nie jest
ograniczony liczbą przykładów.** Jest ograniczony tym, czego o pacjencie nie wie.
Lista braków jest już w `features_notes.md` (`number_diagnoses`, `number_outpatient`,
`A1Cresult`, `admission_type_id`, `discharge_disposition_id`) i to tam, a nie w doborze
modelu czy w hiperparametrach, leży najbliższy realny zysk.

### 4. Dwa progi to dwa różne produkty

| | próg | oflagowanych | precyzja | czułość | co to znaczy operacyjnie |
|---|---:|---:|---:|---:|---|
| `f1` | 0.5052 | 5 256 (37.6%) | 0.124 | 0.518 | łapiesz połowę readmisji, ale koordynator dostaje co trzeci wypis — nierealne |
| `recall_at_precision` | 0.6743 | 511 (3.7%) | 0.198 | 0.080 | wykonalne kadrowo, ale przechwytujesz 8% przypadków |

F1 nie jest „metryką neutralną" — zakłada, że fałszywy alarm kosztuje dokładnie tyle,
co przeoczony pacjent. W readmisjach to nieprawda: fałszywy alarm to jeden telefon
pielęgniarki, a przeoczenie to ponowna hospitalizacja za kilkanaście tysięcy dolarów.
Dlatego `run_experiment` ma domyślnie `criterion="recall_at_precision"` — pojemność zespołu
jest twardym ograniczeniem, a nie zmienną do optymalizacji.

---

## Do poprawy w kodzie

**1. `run_experiment` omija własną walidację.** Buduje model przez
`build_full_pipeline(MODELS[model_name](**params))`, czyli sięga do słownika bezpośrednio,
zamiast przez `train_model`. Nieznana nazwa modelu daje `KeyError` po dwóch minutach
cross-walidacji zamiast czytelnego `ValueError` natychmiast. Do tego konstrukcja pipeline'u
jest zduplikowana w dwóch miejscach.

**2. Brak marginesu w `find_threshold`.** Punkt dokładnie na granicy nie dowozi kontraktu
(dowód wyżej). Warianty: parametr `margin`, albo wybór po dolnym końcu przedziału ufności
precyzji zamiast po punktowej estymacie.

**3. `class_weight="balanced"` w `BASELINE_PARAMS`.** Nie poprawia rankingu (Δpr_auc = 0.0002),
psuje Briera 3×. Skoro `brier` jest w `compute_metrics`, to obecnie mierzysz metrykę,
którą sam świadomie sabotujesz jednym parametrem.

**4. `n_splits` różni się między trace'em (3) a domyślnym `run_experiment` (5).** Progi
z tego dokumentu nie będą identyczne z tym, co wypluje `run_experiment` na domyślnych
ustawieniach. Warto o tym pamiętać przy porównywaniu przebiegów.

**5. `tests/models/` zawiera tylko `__init__.py`.** `evaluate.py` jest napisany dokładnie po to,
żeby dało się go testować ręcznymi tablicami — a testów nie ma. Rzeczy, które wołają o test,
bo są nieoczywiste i ciche, gdy się zepsują: obcięcie `[:-1]`, mapowanie `candidates[...]`
z przestrzeni maski do przestrzeni progów, `zero_division` przy zerze flag oraz `ValueError`,
gdy `min_precision` jest nieosiągalne.

**6. `groups=` to dziś martwy kod.** Nie usuwać — ale test, który *naprawdę* sprawdza rozdział
pacjentów, wymaga kohorty ze `strategy="all"`. Obecny `shared patients: 0` przechodzi zawsze.

---

## Co z tego mówisz na rozmowie

- Dlaczego próg wybierasz na out-of-fold, a nie na teście i nie na predykcjach modelu
  z treningu — i co konkretnie znaczy, że `cross_val_predict` daje predykcje, nie metrykę.
- Że `class_weight="balanced"` i strojenie progu rozwiązują ten sam problem, więc łącząc
  je płacisz kalibracją za nic. Masz na to liczby: Δpr_auc = 0.0002, Brier 0.2379 → 0.0809.
- Że pr_auc trzeba czytać względem odsetka klasy pozytywnej (0.1387 przy bazie 0.0902 =
  1.54×), a nie względem 0.5 jak ROC-AUC.
- Że próg to decyzja biznesowa o pojemności zespołu, wyrażona w `n_flagged`, a nie
  optymalizacja F1.
- Że dowiozłeś dowód, iż więcej danych nie pomaga (79 541 vs 55 992 wierszy, ten sam test,
  zero różnicy) — czyli umiesz odróżnić problem z danymi od problemu z cechami zamiast
  domyślnie dosypywać wierszy.
