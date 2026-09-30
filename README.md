<div dir="rtl">

# 🏘️ پروژه تحلیل داده‌های املاک و سیستم توصیه‌گر (دیوار)

پروژه‌ی تحلیل داده‌های املاک، پیش‌پردازش جامع، تحلیل‌های آماری و توسعه‌ی سیستم توصیه‌گر ملک مبتنی بر خوشه‌بندی.

</div>

---

## 📁 ساختار پروژه (Project Structure)

```text
.
├── data
│   ├── Divar_cleaned.parquet
│   ├── Divar.csv
│   └── iran_city_classification.csv
├── eda
│   ├── eda_01.ipynb
│   ├── eda_02.ipynb
│   ├── eda_03_04_06.ipynb
│   ├── eda_05.ipynb
│   ├── eda_07.ipynb
│   ├── eda_08.ipynb
│   └── eda_09.ipynb
├── hypothesis-test
│   ├── 01_hypothesis_tests.ipynb
│   ├── 02_hypothesis_tests.ipynb
│   └── 03_04_hypothesis_tests.ipynb
├── notebook
│   ├── main_pipeline.ipynb
│   └── notebook-documentation
│       ├── 01a_preprocessing.ipynb
│       ├── 02a_preprocessing.ipynb
│       ├── araz_preprocessing.ipynb
│       ├── boolean_feature.ipynb
│       ├── finance_cleaning.ipynb
│       ├── shahab_preprocessing.ipynb
│       └── time_location_preprocessing.ipynb
├── prediction
│   └── house_price_prediction.ipynb
├── README.md
├── recommender-system
│   └── real_state_clustering.ipynb
├── requirements.txt
└── scripts
    ├── preprocessing_utils.py
    └── __pycache__
        └── preprocessing_utils.cpython-312.pyc

<div dir="rtl">

## 🔄 آخرین تغییرات و وضعیت پروژه (Recent Progress)

جهت هماهنگی اعضای تیم، تغییرات زیر روی ساختار پروژه و خط لوله داده اعمال شده است:

### ۱. یکپارچه‌سازی و ایجاد نوت‌بوک اصلی (`main_pipeline.ipynb`)

- تمامی کدهای متفرقه‌ی پیش‌پردازش و آزمایش‌ها منسجم شده و یک Pipeline شفاف و خطی در نوت‌بوک اصلی `notebook/main_pipeline.ipynb` شکل گرفته است.
- کدهای آزمایشی قدیمی به پوشه `notebook-documentation` منتقل شدند تا نوت‌بوک اصلی خلوت و آماده اجرای مراحل بعدی باشد.

### ۲. ذخیره‌سازی یکپارچه و ساخت دیتای تمیزشده (`Divar_cleaned.parquet`)

- **استفاده از فرمت Parquet به‌جای CSV**: تمامی خروجی‌ها و داده‌های تمیزشده در قالب فایل `Divar_cleaned.parquet` در پوشه `data/` ذخیره می‌شوند. انتخاب فرمت Parquet به دلیل فشرده‌سازی بسیار بالا، حفظ دقیق نوع داده‌ها (`Data Types` نظیر Bool و Int64) و سرعت فوق‌العاده بالا در بارگذاری نسبت به `CSV` انجام شده است.

### ۳. ضرورت اجرای `main_pipeline.ipynb` پیش از هرگونه تحلیل

با توجه به اینکه پیش‌پردازش داده در چند نوت‌بوک جداگانه توسط اعضای مختلف تیم انجام می‌شد و هرکدام بخشی از ستون‌ها را پاک‌سازی می‌کردند، این کدها اکنون در یک تابع واحد و متوالی به نام `run_full_preprocessing_pipeline` (در `scripts/preprocessing_utils.py`) ادغام شده‌اند. این تابع تمام مراحل پیش‌پردازش (دسته‌بندی و موقعیت، ویژگی‌های عددی ساختمان، متن و اجاره روزانه، امکانات، ویژگی‌های مالی، و زمان/مکان) را به‌صورت خطی و به ترتیب صحیح روی داده‌ی خام اجرا می‌کند.

**نکته‌ی مهم و الزامی برای همه‌ی اعضای تیم:**

هرگونه تحلیل آماری، EDA، یا مدل‌سازی باید **صرفاً** روی خروجی `data/Divar_cleaned.parquet` انجام شود، نه روی `Divar.csv` خام یا نسخه‌های موقتی که هرکدام از ما جداگانه پردازش کرده بودیم. برای تولید این فایل:

1. حتماً قبل از شروع کار، از پوشه‌ی `notebook-documentation` بررسی کنید که آخرین تغییرات پیش‌پردازش (اگر بخشی جدید اضافه کرده‌اید) در `scripts/preprocessing_utils.py` merge شده باشد.
2. نوت‌بوک `notebook/main_pipeline.ipynb` را از ابتدا تا انتها اجرا کنید تا `Divar_cleaned.parquet` بازتولید شود.
3. فقط بعد از این مرحله، سراغ سؤالات آماری/تحلیلی یا توسعه‌ی سیستم توصیه‌گر بروید.

اگر نوت‌بوک آزمایشی جدیدی (مثل پیش‌پردازش یک گروه خاص از ستون‌ها) نوشتید، لطفاً پیش از merge کردن با main، اطمینان حاصل کنید که منطق آن با نسخه‌ی فعلی `preprocessing_utils.py` هماهنگ است (چه ستونی را حذف/نگه می‌دارد، نوع داده‌ی خروجی چیست) تا از بازگشت باگ‌های قبلاً رفع‌شده جلوگیری شود.

</div>
