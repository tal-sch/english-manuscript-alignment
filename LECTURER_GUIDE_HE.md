# מדריך להצגת פרויקט Manuscript Alignment

## 1. המשפט שמסביר את כל הפרויקט

המערכת מקבלת שתי תמונות של אותה שורת טקסט: תמונת מקור `Is` ותמונת יעד `It`.
היא מעריכה לכל פיקסל ביעד מאיזו נקודה במקור צריך לדגום, וכך יוצרת את
`Ialigned` — המקור לאחר עיוות גאומטרי למערכת הקואורדינטות של היעד.

המערכת אינה OCR, אינה מחליפה סגנון כתב ואינה מייצרת אותיות חדשות. היא לומדת
registration גאומטרי מתוך image patches, ולכן אינה תלויה בכך שהמילים הופיעו
באימון.

## 2. ההגדרה המתמטית

המודל משתמש ב-backward flow. לכל פיקסל יעד `x=(x,y)` הוא מנבא וקטור
`u(x)=(ux,uy)` שמצביע למיקום שיש לדגום בתמונת המקור:

```text
Ialigned(x) = Is(x + u(x))
```

הדגימה נעשית באמצעות `torch.grid_sample` עם אינטרפולציה בילינארית, ולכן הפעולה
דיפרנציאבילית ואפשר להעביר דרכה gradients בזמן האימון.

## 3. מדוע לא להסתפק בטרנספורמציה אפינית

Affine יחיד מטפל בהזזה, scale, rotation ו-shear גלובליים. הוא אינו יכול לתקן
רווח שונה בין שתי מילים, מילה אחת רחבה יותר, או baseline שמתעקם מקומית. לכן
המערכת היא היברידית:

1. שלב affine דטרמיניסטי מיישר את גבולות הדיו ומסיר scale/translation גדולים.
2. רשת dense registration מנבאת תיקון מקומי וחלק.

שלב ה-affine הנוכחי מעריך scale ו-translation בצירי x/y מתוך bounding box של
הדיו. הוא אינו מעריך rotation או shear מפורשים; תיקונים קטנים כאלה מיוצגים
על ידי ה-dense flow.

## 4. מבנה המודל

קלט האימון הוא `B x 1 x 96 x 512`.

### Shared feature pyramid

אותו encoder, עם אותם משקלים, מעבד את המקור ואת היעד:

```text
96x512x1
 -> 48x256x32
 -> 24x128x64
 -> 12x64x96
```

השיתוף חשוב: descriptors של המקור ושל היעד נמצאים באותו מרחב תכונות.

### Horizontal patch correlation

במפה העמוקה מבצעים ממוצע על הגובה. כל עמודת feature map הופכת ל-descriptor של
בלוק אנכי חופף בשורת הטקסט. לאחר L2 normalization מחשבים cosine similarity:

```text
C(t,s) = cosine(Ft(t), Fs(s))
```

מתקבלת מטריצת correlation בגודל `64 x 64`: לכל בלוק יעד, כמה הוא דומה לכל
בלוק מקור. softmax ו-expected source position נותנים coarse horizontal flow.
positional prior מעדיף התאמות קרובות אך אינו אוסר הזזה.

### Dense residual decoder

תכונות המקור מעוותות תחילה בעזרת ה-coarse flow. decoder בסגנון U-Net משלב:

- warped source features;
- target features;
- skip features ברזולוציות שונות;
- coarse flow;
- confidence של ההתאמה.

ה-decoder מחזיר שני ערוצים לכל פיקסל: displacement אופקי ואנכי. התיקון מוגבל
באימון באמצעות `tanh` ל-48 פיקסלים, ומחובר ל-coarse flow.

למודל 1,117,219 פרמטרים.

## 5. יצירת זוגות האימון

### IAM

`data.py` קורא תמונות forms ואת קובצי ה-XML, מאחד את רכיבי `cmp` ל-bounding box
של שורה, וחותך את השורה מהטופס.

הפיצול נעשה לפי `writer-id`, לא באקראי לפי תמונה:

| Split | Lines | Writers | OOV words |
|---|---:|---:|---:|
| Train | 8,101 | 459 | - |
| Validation | 1,460 | 98 | 13.45% |
| Test | 1,783 | 100 | 13.54% |

משורת IAM אמיתית נוצרים source ו-target באמצעות flow ידוע שמכיל affine ו-elastic
deformation. מכיוון שאנחנו יצרנו את ה-flow, יש dense ground truth מדויק.

### Cross-font synthetic data

המילים מחולקות לפני rendering ל-70/15/15, ולכן מילות test אינן מופיעות באימון.
אותו רצף מילים מצויר בשני פונטים שונים בתוך semantic word cells משותפים. פונט
Satisfy מוחזק מחוץ לאימון.

בזוג כזה glyphs שונים, ולכן אין הצדקה ל-pixel loss. ה-flow הגאומטרי הידוע עדיין
משמש supervision, אבל ה-photometric mask הוא אפס.

### Identity pairs

ב-fine-tuning, 20% מזוגות IAM מקבלים flow אפס. כך המודל לומד שלא לעוות תמונות
שכבר מיושרות.

## 6. פונקציית ההפסד

```text
L = 1.00 Lflow
  + 0.35 Lcoarse
  + 0.50 Lphoto
  + 0.25 Lssim
  + 0.03 Lsmooth
  + 0.10 Lmonotonic
```

- `Lflow`: שגיאת L1 מול dense ground-truth flow.
- `Lcoarse`: supervision לרכיב האופקי של ה-coarse correlation flow.
- `Lphoto`: Charbonnier distance בין aligned ל-target, עם משקל גבוה יותר לדיו.
- `Lssim`: שומר על מבנה מקומי ולא רק על ערכי פיקסלים.
- `Lsmooth`: מעניש שינויים חדים בין וקטורי flow שכנים.
- `Lmonotonic`: מעניש מצב שבו `x + ux(x)` מפסיק לעלות, כדי לצמצם folding.

ה-monotonicity הוא penalty רך ולא הוכחה מתמטית שה-flow לעולם לא יתקפל. שלב
ה-affine החדש מסיר את השינויים הגלובליים הגדולים שהיו גורמים מרכזיים לקיפול.

## 7. תהליך האימון

1. `train_registration.py` טוען את IAM ומייצר writer-disjoint split.
2. הוא מחבר 8,101 דוגמאות IAM עם 4,000 דוגמאות cross-font.
3. DataLoader מחזיר source, target, ground-truth flow ומסכות.
4. המודל מחזיר `aligned`, `flow`, `coarse_flow`, `confidence`, `similarity`.
5. `RegistrationLoss` מחשב את כל רכיבי ההפסד.
6. AdamW מעדכן משקלים; CosineAnnealing מוריד learning rate.
7. checkpoint נשמר לפי validation EPE הנמוך ביותר.

האימון המרכזי היה 30 epochs, batch size 32, learning rate `2e-4`, BF16 AMP.
לאחריו בוצעו 8 epochs של identity fine-tuning ב-learning rate `5e-5`.

## 8. תהליך inference מלא

1. המשתמש מעלה Source ו-Target שמכילים אותו טקסט.
2. התמונות מומרות ל-grayscale ועוברות autocontrast.
3. הן נמדדות לגובה 96 תוך שמירת aspect ratio וממוקמות על canvas משותף.
4. מחושב bounding box של הדיו בכל תמונה.
5. affine backward flow ממפה את גבולות דיו היעד לגבולות דיו המקור.
6. נוצר affine-prealigned source לצורך קלט לרשת.
7. אם השורה רחבה מ-512, היא מחולקת לחלונות 512 עם 50% overlap.
8. המודל מנבא residual dense flow בכל חלון.
9. Hann weights מחברים את ה-flows החופפים בלי seam חד.
10. מרכיבים את ה-affine flow ואת ה-residual flow:

```text
u(x) = ur(x) + ua(x + ur(x))
```

11. מפעילים warp אחד בלבד על המקור המקורי בעזרת ה-flow המורכב.
12. האתר מציג affine result, final result, overlays, flow ו-confidence.

ה-warp היחיד חשוב: שתי פעולות resize/warp רצופות היו יוצרות טשטוש כפול.

## 9. הערכה ותוצאות

המדד המרכזי הוא Endpoint Error:

```text
EPE = mean(||upred - ugt||2)
```

מדדים נוספים: אחוז פיקסלים מתחת ל-1/3/5 פיקסלים, MAE, SSIM ו-Ink Dice.

| Test | Identity | Selected model |
|---|---:|---:|
| IAM EPE | 26.54 | 5.47 |
| Cross-font EPE | 26.23 | 6.56 |
| Exact-pair motion | 0.00 | 0.32 |
| Real cross-writer landmark error | 26.50 | 10.34 |

ב-real-pair evaluation נמצאו עשר שורות test זהות שנכתבו על ידי כותבים שונים.
מרכזי המילים מה-XML משמשים landmarks אמיתיים. המודל הוריד את השגיאה ב-60.98%.

## 10. תפקיד הקבצים

### הליבה

| File | Job |
|---|---|
| `manuscript_registration/model.py` | encoder, patch correlation, decoder ויצירת dense flow |
| `manuscript_registration/geometry.py` | המרה בין flow ל-grid, warp, resize ויצירת flow סינתטי |
| `manuscript_registration/data.py` | IAM XML, writer split, augmentations, IAM ו-cross-font datasets |
| `manuscript_registration/losses.py` | כל רכיבי פונקציית ההפסד |
| `manuscript_registration/metrics.py` | EPE, pixel accuracy, MAE, SSIM, Dice ו-identity baseline |
| `manuscript_registration/inference.py` | normalization, affine prealignment, tiled inference, flow composition ו-visualization helpers |
| `manuscript_registration/real_pairs.py` | איתור שורות זהות מכותבים שונים ומיפוי word landmarks |

### אימון והערכה

| File | Job |
|---|---|
| `train_registration.py` | orchestration מלא של האימון, validation ו-checkpoints |
| `evaluate_registration.py` | הערכת IAM writer-disjoint עם flow ידוע |
| `evaluate_cross_font.py` | מילים ופונט שלא הופיעו באימון |
| `evaluate_identity.py` | בדיקת source=target ו-flow רצוי אפס |
| `evaluate_real_pairs.py` | word-landmark evaluation על כתבי יד אמיתיים |
| `visualize_registration.py` | panels איכותניים של source/target/aligned/overlay/flow |

### שימוש והצגה

| File | Job |
|---|---|
| `registration_web_app.py` | אתר Gradio להעלאת שתי תמונות ולהצגת כל שלבי היישור |
| `align_images.py` | inference משורת הפקודה ושמירת קובצי הפלט |
| `tests/test_registration.py` | 12 בדיקות ל-flow conventions, datasets, model, tiling, affine ו-composition |
| `README.md` | הוראות התקנה, אימון, הערכה ושימוש |
| `PROJECT_REPORT.md` | הניסוח האקדמי, התוצאות, ablations והמגבלות |

### קבצי legacy

`prepare_yolo_dataset.py`, `train_yolo.py`, `train_siamese_triplet.py` ו-
`english_alignment_web_app.py` הם המערכת הישנה: YOLO + Siamese + Smith-Waterman.
היא מוצאת התאמות בין מילים אך אינה מחזירה `Ialigned`, ולכן היא baseline ולא
הפתרון הראשי.

## 11. תשובת הצגה קצרה למרצה

"התחלנו ממערכת שמזהה מילים ומתאימה רצפים, אבל היא הייתה תלויה בזהויות מילים ולא
יצרה תמונה מיושרת. החלפנו אותה ברשת registration fully convolutional. שני הקלטים
עוברים shared encoder, וכל עמודת feature map משמשת patch descriptor. correlation
גלובלי מספק התאמה אופקית גסה, ו-decoder מנבא dense 2D residual flow. spatial
transformer דיפרנציאבילי דוגם את המקור ומחזיר את `Ialigned`. האימון משתמש ב-IAM
עם writer-disjoint split, warps בעלי ground truth, מילים ופונט מוחזקים מחוץ
לאימון, ו-identity pairs. בזמן inference אנו מסירים תחילה scale ו-translation
גלובליים באמצעות affine ink alignment, מעבדים שורות ארוכות בבלוקים חופפים,
מרכיבים את ה-flow האפיני והעצבי ומפעילים warp יחיד על המקור."

## 12. מגבלות שצריך לומר ביושר

- שתי התמונות חייבות להכיל אותו טקסט ובאותו סדר.
- מילים חסרות או נוספות דורשות visibility/unmatched mask שאינו קיים כרגע.
- הבדל סגנון אינו style transfer; גם יישור נכון לא יוצר pixel overlap מושלם.
- ה-affine preprocessing הנוכחי אינו מעריך rotation/shear מפורשים.
- monotonicity הוא penalty רך; hard monotonic parameterization הוא שיפור עתידי.
- scale קיצוני, חיתוך טקסט או רזולוציה שאיבדה strokes אינם ניתנים לתיקון מלא.
