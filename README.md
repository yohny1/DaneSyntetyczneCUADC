# Generator danych syntetycznych — Strike and Reconnaissance (fixed wing)

Dwa skrypty:
- `generato2.py` — generuje syntetyczny dataset YOLO (obrazy + etykiety)
- `viz_yolo.py` — rysuje bboxy z etykiet na obrazach, żeby sprawdzić poprawność datasetu

---

## 1. Przygotowanie danych wejściowych

| Folder | Co ma zawierać | Wymagane? |
|---|---|---|
| `targets/` (`--targets-dir`) | Sylwetki celów obrazkowych — PNG/JPG, nazwa pliku bez rozszerzenia = nazwa klasy (np. `mannequin.png`, `tent.png`) | Tak, jeśli chcesz cele obrazkowe |
| `backgrounds/` (`--backgrounds-dir`) | Twoje własne zdjęcia teł (np. lotnicze/aerial) | Nie — bez tego generator używa teł proceduralnych (trawa/piach/susza) |
| `distractors/` (`--distractors-dir`) | Dodatkowe obiekty-śmieci wklejane w tło (np. kamienie, śmieci, inne obiekty) | Nie |

Sylwetki w `targets/` muszą być **ciemne na jasnym tle** (albo mieć kanał alfa) — generator binaryzuje je progiem `gray < 128`.

---

## 2. Generowanie datasetu

```bash
python generato2.py \
  --targets-dir targets \
  --backgrounds-dir backgrounds \
  --distractors-dir distractors \
  --out-dir dataset \
  --num-images 3000 \
  --img-w 1280 --img-h 720 \
  --mode mixed \
  --max-scale 0.16
```

### Najważniejsze parametry

| Parametr | Domyślnie | Znaczenie |
|---|---|---|
| `--mode` | `mixed` | `mixed` = losowo cyfry/obrazki, `digits` = tylko panele z cyframi, `images` = tylko panele z sylwetkami |
| `--min-scale` / `--max-scale` | `0.06` / `0.30` | Zakres wielkości całego panelu (strzałki) jako ułamek wysokości kadru. **Obniż `--max-scale`, jeśli cele są za duże** |
| `--max-targets` | `3` | Ile paneli maksymalnie na jednym obrazie |
| `--max-distractors` | `4` | Ile przeszkadzaczy (kamienie/kije/papier/puste kołki/custom) na obrazie |
| `--arrow-rot` | `360` | Zakres losowego obrotu kierunku strzałki [°] |
| `--max-rot` / `--persp` | `20` / `0.15` | Losowa rotacja i zniekształcenie perspektywiczne przy wklejaniu panelu w scenę |
| `--digits` | `0123456789` | Które cyfry generować jako klasy |
| `--train-ratio` | `0.9` | Podział train/val |
| `--seed` | `42` | Ziarno losowości — ten sam seed = ten sam dataset |

### Co powstaje

```
dataset/
├── images/{train,val}/000000.jpg ...
├── labels/{train,val}/000000.txt ...
└── data.yaml
```

Każda linia w `.txt` to standardowy format YOLO:
```
<class_id> <x_center> <y_center> <width> <height>   # znormalizowane 0–1
```

### Struktura klas (kolejność = kolejność indeksów w etykietach)

```
0..9              -> cyfry '0'-'9' (kolejność wg --digits)
kolejne indeksy   -> obiekty z targets/ (posortowane alfabetycznie po nazwie pliku)
przedostatni      -> "panel"   — bbox całej płachty/strzałki
ostatni           -> "empty"   — panel bez treści (pusty target)
```

**Dokładną listę z numerami** znajdziesz zawsze w `dataset/data.yaml` (pole `names:`) — to samo wypisuje się też w konsoli przy starcie skryptu. Nie zgaduj numerów ręcznie, tylko czytaj z tego pliku.

> ⚠️ `data.yaml` zapisuje się (obecnie) dopiero **po zakończeniu całej generacji**. Jeśli przerwiesz skrypt w trakcie, pliku nie będzie, mimo że obrazy/etykiety już częściowo istnieją — numerację klas i tak zobaczysz w logu konsoli na starcie.

Każdy panel z treścią ma **dwa zagnieżdżone bboxy**: `panel` (cała płachta) + klasa symbolu (cyfra/obiekt) w środku. Panel bez treści ma tylko jeden bbox z klasą `empty`.

---

## 3. Wizualizacja / kontrola jakości

```bash
python viz_yolo.py --dataset ./dataset --split train --num 30 --output ./viz_check
```

| Parametr | Znaczenie |
|---|---|
| `--dataset` | Folder datasetu (ten sam co `--out-dir` generatora) |
| `--split` | `train` lub `val` |
| `--num` | Ile losowych obrazów zwizualizować |
| `--output` | Gdzie zapisać obrazy z narysowanymi bboxami |

Wynik: obrazy w `--output` z narysowanymi ramkami i nazwami klas — sprawdź tu, zanim zaczniesz trening, czy panele/cele/empty są oznaczone poprawnie i czy rozmiary celów są sensowne.
