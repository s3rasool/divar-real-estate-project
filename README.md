# پروژه تحلیل داده املاک دیوار

این پروژه یک Workflow کامل Data Science روی دیتاست بزرگ آگهی‌های املاک **دیوار** است.

بخش‌های اصلی پروژه شامل:

- Preprocessing
- Feature Engineering
- Exploratory Data Analysis
- Hypothesis Testing
- Clustering
- House Price Prediction

است.

---

## معرفی پروژه

دیتاست خام شامل حدود **یک میلیون آگهی املاک** است و اطلاعات متنوعی از جمله موارد زیر را در بر می‌گیرد:

- املاک مسکونی و تجاری
- آگهی‌های فروش و اجاره
- مشخصات فیزیکی ملک
- اطلاعات مالی
- Featureهای دسته‌ای
- مختصات جغرافیایی
- عنوان و توضیحات آگهی
- امکانات ملک
- اطلاعات زمانی

هدف اصلی پروژه این است که داده خام و ناهمگون دیوار را به یک دیتاست تمیز و قابل تحلیل تبدیل کنیم و سپس از آن برای تحلیل آماری، Clustering و Prediction استفاده کنیم.

خروجی نهایی Pipeline بخش Preprocessing:

- **999,997 ردیف**
- **79 Feature پاک‌سازی‌شده یا مهندسی‌شده**
- فایل خروجی: `Divar_cleaned.parquet`

---

# روند کلی پروژه

```text
Raw Divar Dataset
        │
        ▼
Preprocessing
        │
        ├── Data Type Normalization
        ├── Missing Value Analysis
        ├── Financial Feature Cleaning
        ├── Boolean / Categorical Cleaning
        ├── Text Normalization
        ├── Time Feature Engineering
        └── Geographic Validation
        │
        ▼
Divar_cleaned.parquet
        │
        ├──────────────► EDA
        │
        ├──────────────► Hypothesis Testing
        │
        ├──────────────► Clustering
        │
        └──────────────► House Price Prediction
```

---

# 1. Preprocessing

منطق اصلی Preprocessing در این پروژه این بود که داده‌ها به‌صورت کورکورانه حذف یا Impute نشوند.

اصل کلی پروژه:

> **قبل از حذف، جایگزینی یا اصلاح یک مقدار، ابتدا باید بررسی شود که آیا آن مقدار Missing یا غیرعادی واقعاً خطای داده است یا یک حالت ساختاری طبیعی.**

Notebookهای مختلف Preprocessing در نهایت در یک Pipeline اصلی ادغام شدند.

---

## Featureهای دسته‌ای

ستون‌هایی مانند:

- `cat2_slug`
- `cat3_slug`
- `city_slug`
- `neighborhood_slug`
- `user_type`
- `deed_type`
- `building_direction`
- `floor_material`

استانداردسازی شدند.

مراحل اصلی شامل:

- حذف فاصله‌های اضافی
- یکسان‌سازی حروف فارسی و عربی
- تبدیل مقادیر نامعتبر مانند `unselect`
- حفظ Missingهای ساختاری در موارد لازم

بود.

---

## Featureهای عددی ملک

Featureهایی مانند:

- `building_size`
- `construction_year`
- `floor`
- `rooms_count`
- `total_floors_count`
- `unit_per_floor`

به قالب عددی استاندارد تبدیل شدند.

نمونه عملیات انجام‌شده:

- تبدیل اعداد فارسی و عربی
- تبدیل تعداد اتاق‌های متنی به عدد
- تبدیل مقادیری مانند `30+`
- بررسی محدوده‌های غیرمنطقی متراژ

ستون `land_size` Missing زیادی داشت، بنابراین به‌جای اتکا مستقیم به آن، Feature جدیدی با نام:

```text
has_land
```

ساخته شد.

---

## Boolean Featureها و امکانات

Featureهای Boolean به سه حالت استاندارد تبدیل شدند:

```text
True
False
<NA>
```

این کار باعث می‌شود تفاوت بین موارد زیر حفظ شود:

- Feature وجود دارد
- Feature وجود ندارد
- اطلاعات Feature اصلاً ثبت نشده است

نکته مهم:

```text
Missing != False
```

یعنی Missing بودن یک Feature الزاماً به معنی نبود آن ویژگی نیست.

---

## Featureهای مالی

ستون‌های مربوط به:

- Price
- Credit
- Rent
- Transform

به‌صورت جداگانه بررسی شدند.

در این بخش:

- Boolean Flagهای مالی استاندارد شدند
- Missingهای ساختاری حفظ شدند
- Full Credit شناسایی شد
- Feature جدید `inferred_full_credit` ساخته شد
- مقادیر مشکوک Rent بررسی شدند
- تا جای ممکن ستون‌های خام بدون Overwrite نگه داشته شدند

هدف این بود که اطلاعات مالی اصلی از بین نرود و Featureهای جدید در کنار داده خام ساخته شوند.

---

## Text Features

ستون‌های:

```text
title
description
```

نرمال‌سازی شدند.

مراحل شامل:

- تبدیل `ي` عربی به `ی`
- تبدیل `ك` عربی به `ک`
- حذف فاصله‌های اضافی
- تبدیل متن خالی به Missing

Featureهای جدید:

- `title_length`
- `description_length`
- `title_word_count`
- `description_word_count`

نیز ساخته شدند.

Text خام مستقیماً وارد مدل Prediction نشد، چون ممکن بود قیمت داخل توضیحات آگهی ذکر شده باشد و Target Leakage ایجاد کند.

---

## Featureهای زمانی

از تاریخ آگهی Featureهایی مانند:

- سال شمسی
- ماه شمسی
- `month_index`

استخراج شد.

این Featureها برای تحلیل Trend و Seasonality بازار استفاده شدند.

---

## Featureهای جغرافیایی

مختصات جغرافیایی با چند روش بررسی شدند:

- محدوده جغرافیایی تقریبی ایران
- Coordinateهای پرتکرار
- Coordinate مشترک بین چند شهر
- فاصله از مرکز تقریبی شهر

در نهایت حدود **636 هزار آگهی** دارای Coordinate معتبر برای تحلیل جغرافیایی بودند.

بعضی Featureهای جغرافیایی برای مدل Prediction استفاده نشدند، زیرا قبل از Train/Test Split و بر اساس کل دیتاست ساخته شده بودند و احتمال Leakage وجود داشت.

---

# 2. Exploratory Data Analysis

EDA روی فایل:

```text
Divar_cleaned.parquet
```

انجام شد.

هدف این بخش شناخت ساختار داده بدون تغییر مستقیم در دیتاست اصلی بود.

---

## بررسی Categoryها

توزیع Featureهای:

```text
cat2_slug
cat3_slug
```

بررسی شد.

بزرگ‌ترین گروه‌های دیتاست:

- `residential-sell`
- `residential-rent`

بودند.

---

## بررسی سال ساخت

Distribution ستون:

```text
construction_year
```

بررسی شد تا ساختار سن ساختمان‌ها و تمرکز داده در سال‌های مختلف مشخص شود.

---

## بررسی روند ماهانه فروش و اجاره

Feature تحلیلی جدیدی با نام:

```text
listing_type
```

ساخته شد.

دسته‌های فروش به:

```text
sale
```

و دسته‌های اجاره به:

```text
rent
```

تبدیل شدند.

سپس تعداد آگهی‌ها در ماه‌های مختلف بررسی شد.

---

## تحلیل Price

توزیع Price برای Property Typeهای مختلف با استفاده از:

- Histogram
- Log Scale
- Boxplot
- Percentile Filter برای Visualization

بررسی شد.

توزیع قیمت‌ها به‌شدت Right-Skewed بود.

---

## تحلیل جغرافیایی

تراکم آگهی‌ها روی نقشه با استفاده از Coordinateهای معتبر بررسی شد.

Pinهای بسیار پرتکرار در بعضی تحلیل‌ها کنار گذاشته شدند تا Hotspot مصنوعی ایجاد نشود.

---

## Trend اجاره

روند Rent به‌صورت ماهانه بررسی شد.

برای این تحلیل:

- Missingها
- Rent صفر
- Extreme Upper Values

فیلتر شدند.

---

## تحلیل قیمت با تعدیل تورم

Price اسمی سال‌های مختلف با Price تعدیل‌شده با تورم مقایسه شد.

به دلیل کم بودن Sample بعضی سال‌های قدیمی، نتایج آن سال‌ها با احتیاط تفسیر شدند.

---

## تحلیل امکانات ملک

Amenityهایی مانند:

- Elevator
- Balcony
- Pool
- Barbecue
- Security Guard

بررسی شدند.

در مقایسه شهرها از درصد استفاده شد، نه صرفاً تعداد خام.

---

# 3. Hypothesis Testing

چهار سؤال اصلی آماری بررسی شدند.

---

## 3.1 آیا خانه‌ها در کلان‌شهرها کوچک‌تر هستند؟

پس از فیلتر متراژهای Extreme:

- Mean متراژ کلان‌شهرها: حدود **120.4 متر**
- Mean شهرهای کوچک‌تر: حدود **136.3 متر**

Welch's T-Test و Mann-Whitney U هر دو شواهد آماری قوی از وجود تفاوت نشان دادند.

---

## 3.2 آیا خانه‌های قدیمی بزرگ‌تر هستند؟

خانه‌های قبل از سال 1396 با خانه‌های جدیدتر مقایسه شدند.

Mean مشاهده‌شده:

- خانه‌های قدیمی: حدود **106.6 متر**
- خانه‌های جدید: حدود **126 متر**

بنابراین فرض اولیه که خانه‌های قدیمی بزرگ‌تر هستند توسط داده پشتیبانی نشد.

---

## 3.3 آیا Business Deed با Price ملک تجاری ارتباط دارد؟

املاک تجاری دارای Business Deed با املاک بدون آن مقایسه شدند.

در کل Dataset ارتباط آماری معناداری دیده شد، اما مقدار اثر بین:

- شهرها
- نوع ملک تجاری

متفاوت بود.

بنابراین این نتیجه باید به‌عنوان:

```text
Association
```

تفسیر شود، نه رابطه علّی.

---

## 3.4 آیا Amenityها با قیمت ویلا ارتباط دارند؟

Luxury Amenityهایی مانند:

- Pool
- Jacuzzi
- Sauna
- Barbecue

با Price مقایسه شدند.

در بسیاری از موارد ارتباط مثبت مشاهده شد.

با این حال Featureهایی مانند:

- Location
- Building Size
- Land Size

می‌توانند Confounder باشند.

---

# 4. Clustering

دو الگوریتم برای Clustering استفاده شدند:

- K-Means
- DBSCAN

---

## K-Means

Featureهای اصلی:

- Unified Price
- UTM X
- UTM Y
- Building Size

بودند.

Price و Building Size ابتدا Log Transform شدند.

سپس Featureها با:

```text
StandardScaler
```

استاندارد شدند.

اجرای اولیه K-Means با:

```text
k = 10
```

انجام شد.

برای انتخاب تعداد مناسب Cluster از:

- WCSS
- Elbow Method
- KneeLocator
- Silhouette Score

استفاده شد.

KneeLocator مقدار:

```text
k = 6
```

را پیشنهاد داد.

نکته:

Visualization و Cluster Centerهای اصلی Notebook مربوط به اجرای اولیه `k=10` هستند، ولی نتیجه تحلیلی بعدی `k=6` بوده است.

---

## DBSCAN

برای DBSCAN از:

- Location
- Log Price

استفاده شد.

به دلیل هزینه محاسباتی بالا، یک Sample تصادفی با اندازه:

```text
50,000
```

انتخاب شد.

پارامترهای نهایی:

```text
eps = 0.507
min_samples = 20
```

نتایج:

| Metric | مقدار |
|---|---:|
| Silhouette Score | **0.473** |
| Davies-Bouldin Index | **0.890** |
| Noise Ratio | **حدود 0.4٪** |

در Grid Search فقط حالت‌هایی بررسی شدند که دقیقاً 3 Cluster تولید می‌کردند.

بنابراین نتیجه DBSCAN را نباید به‌عنوان جستجوی آزاد بهترین تعداد Cluster تفسیر کرد.

---

# 5. House Price Prediction

هدف بخش Prediction ساخت مدل Regression برای پیش‌بینی ارزش مالی آگهی‌های فروش و اجاره بود.

---

## ساخت Target

چهار نوع معامله وارد مدل شدند:

- `residential-sell`
- `commercial-sell`
- `residential-rent`
- `commercial-rent`

`temporary-rent` به دلیل نبود اطلاعات مالی کافی حذف شد.

برای Sale:

```text
target = price_value
```

برای Rent:

```text
target = credit_value + rent_value × conversion_rate
```

Conversion Rate از خود Dataset استخراج شد.

Median Rate:

```text
33.3333
```

Coverage Target از:

```text
71.63%
```

به:

```text
96.39%
```

افزایش پیدا کرد.

---

## کنترل کیفیت Target

Outlierهای آماری روی Log Target با:

```text
3 × IQR
```

شناسایی شدند.

همچنین بعضی Targetهای مشکوک با Ruleهای مشخص حذف شدند، مانند:

- Sale Price کمتر از 100 میلیون
- Priceهایی با تکرار غیرطبیعی یک رقم

در نهایت:

```text
893,259
```

نمونه برای Modeling باقی ماند.

---

## جلوگیری از Target Leakage

ستون‌هایی که مستقیماً در ساخت Target استفاده شده بودند از Featureهای مدل حذف شدند.

مثلاً:

- `price_value`
- `credit_value`
- `rent_value`
- ستون‌های Transform مالی

همچنین بعضی Featureهای Geographic که قبل از Split ساخته شده بودند نیز حذف شدند.

---

## Featureهای نهایی

مدل از:

```text
34 Raw Features
```

استفاده کرد.

بعد از Preprocessing این Featureها به:

```text
99 Features
```

تبدیل شدند.

---

## Encoding

Featureهای High Cardinality:

- `city_slug`
- `neighborhood_slug`

با Target Encoding تبدیل شدند.

Featureهای دسته‌ای دیگر با One-Hot Encoding تبدیل شدند.

اگر همه Categoryها One-Hot می‌شدند، حدود:

```text
1684 Features
```

تولید می‌شد.

روش Hybrid باعث شد خروجی نهایی فقط 99 Feature باشد.

---

## Train / Validation / Test Split

تقسیم داده:

| بخش | تعداد | سهم |
|---|---:|---:|
| Train | 625,281 | 70٪ |
| Validation | 133,989 | 15٪ |
| Test | 133,989 | 15٪ |

Split با Stratification بر اساس Transaction Type انجام شد.

---

## مدل‌های بررسی‌شده

مدل‌های زیر مقایسه شدند:

- Dummy Baseline
- Linear Regression
- Decision Tree
- Random Forest

نتایج Validation:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Median Baseline | 1.1392 | 1.4556 | -0.0007 |
| Linear Regression | 0.6321 | 0.9570 | 0.5674 |
| Decision Tree | 0.5233 | 0.8653 | 0.6463 |
| Random Forest | **0.5000** | **0.8318** | **0.6732** |

Metricها روی Log Target محاسبه شده‌اند.

---

## Tuning مدل Random Forest

Configuration نهایی:

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    max_features="sqrt",
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)
```

مدل 200 Tree کمی R² بهتری داشت، اما Training Time تقریباً دو برابر شد.

بنابراین مدل 100 Tree به‌عنوان Trade-off بهتر بین Performance و Cost انتخاب شد.

---

## نتیجه نهایی روی Test

| Metric | مقدار |
|---|---:|
| MAE — Log Scale | **0.4904** |
| RMSE — Log Scale | **0.8433** |
| R² — Log Scale | **0.6698** |

Validation R²:

```text
0.6826
```

Test R²:

```text
0.6698
```

اختلاف کم بین Validation و Test نشان می‌دهد Performance مدل روی داده ندیده‌شده نزدیک به Validation باقی مانده است.

نکته مهم:

> `R² = 0.6698` یعنی مدل حدود 66.98٪ Variation در Log Target را توضیح می‌دهد.

این عدد به معنی درست پیش‌بینی شدن 66.98٪ قیمت‌ها نیست.

---

# ساختار Repository

ساختار کلی پروژه:

```text
divar-real-estate-project/
│
├── data/
│   ├── Divar.csv
│   └── Divar_cleaned.parquet
│
├── notebook/
│   ├── main_pipeline.ipynb
│   ├── finance_cleaning.ipynb
│   ├── boolean_feature.ipynb
│   ├── time_location_preprocessing.ipynb
│   └── ...
│
├── prediction/
│   └── house_price_prediction.ipynb
│
├── scripts/
│   └── preprocessing_utils.py
│
├── README.md
└── .gitignore
```

فایل‌های بزرگ Dataset داخل Git قرار نمی‌گیرند و توسط `.gitignore` نادیده گرفته می‌شوند.

---

# نصب Dependencyها

برای نصب Libraryهای اصلی:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn pyarrow kneed jdatetime utm folium
```

نسخه پیشنهادی Python:

```text
Python 3.10+
```

---

# اجرای پروژه

## مرحله 1 — اضافه کردن Dataset

فایل خام را در مسیر زیر قرار دهید:

```text
data/Divar.csv
```

Datasetهای بزرگ داخل Git نگهداری نمی‌شوند.

---

## مرحله 2 — اجرای Preprocessing

Pipeline اصلی از طریق:

```text
notebook/main_pipeline.ipynb
```

اجرا می‌شود.

خروجی:

```text
data/Divar_cleaned.parquet
```

خواهد بود.

---

## مرحله 3 — اجرای تحلیل‌ها

بعد از Preprocessing می‌توان Notebookهای مربوط به:

- EDA
- Hypothesis Testing
- Clustering
- Prediction

را به‌صورت مستقل اجرا کرد.

---

# Libraryها و ابزارهای اصلی

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- PyArrow
- Kneed
- Jdatetime
- UTM
- Folium
- Jupyter Notebook

---

# نکات فنی مهم

## Structural Missingness

Missing Value همیشه به معنی خطا نیست.

مثلاً:

```text
Missing Amenity != Amenity Does Not Exist
```

بعضی Missingها به دلیل متفاوت بودن فرم آگهی بین Categoryها ایجاد شده‌اند.

---

## Data Leakage

در بخش Prediction توجه ویژه‌ای به Leakage شده است.

اقدامات انجام‌شده:

- حذف Featureهای مشتق‌شده از Target
- Fit کردن Preprocessing بر اساس Train در مراحل Modeling
- حذف بعضی Geographic Featureهای ساخته‌شده روی کل Dataset

---

## Extreme Values

Extreme Valueها به‌صورت خودکار حذف نشدند.

در پروژه بین دو مفهوم تفاوت گذاشته شد:

```text
Statistical Outlier
```

و:

```text
Invalid / Suspicious Data
```

فقط زمانی داده حذف شد که دلیل مشخصی برای مشکوک بودن آن وجود داشت.

---

## Target Capping

در Notebook Prediction مرحله‌ای برای Capping Target برنامه‌ریزی شده بود.

اما در نسخه فعلی اجرا، Capping نهایی انجام نشده است.

بنابراین Extreme Priceها همچنان می‌توانند روی Metricهای Raw Toman اثر زیادی داشته باشند.

---

## تفاوت K-Means و DBSCAN

Featureهای ورودی K-Means و DBSCAN کاملاً یکسان نیستند.

K-Means:

```text
Price + Location + Building Size
```

DBSCAN:

```text
Location + Log Price
```

بنابراین Clusterهای این دو الگوریتم نباید مستقیماً معادل یکدیگر تفسیر شوند.

---

# خلاصه نتایج پروژه

```text
Cleaned Dataset:
999,997 Rows × 79 Features

Valid Geographic Coordinates:
~636,000

K-Means Recommended k:
6

DBSCAN:
Silhouette = 0.473
Davies-Bouldin = 0.890
Noise ≈ 0.4%

Prediction Modeling Dataset:
893,259 Rows

Prediction Features:
34 Raw Features
→
99 Processed Features

Final Model:
Random Forest

Final Test R²:
0.6698 — Log Target

Final Test MAE:
0.4904 — Log Target
```

---

# فلسفه کلی پروژه

این پروژه بر چند اصل اصلی بنا شده است:

1. قبل از تغییر یک مقدار، معنای آن بررسی شود.
2. داده خام تا جای ممکن حفظ شود.
3. Missing ساختاری از Data Error جدا شود.
4. Feature جدید به‌جای Overwrite مخرب ساخته شود.
5. از Data Leakage جلوگیری شود.
6. مدل روی داده ندیده‌شده ارزیابی شود.
7. Statistical Significance با Causality اشتباه گرفته نشود.

---

## توضیح نهایی

این پروژه با هدف آموزشی و Data Science روی داده‌های آگهی‌های املاک انجام شده است.

نتایج مدل‌ها و تحلیل‌های آماری نباید به‌عنوان ارزش‌گذاری حرفه‌ای ملک یا توصیه مالی در نظر گرفته شوند.
