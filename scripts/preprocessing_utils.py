import pandas as pd
import numpy as np


def preprocess_category_location_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    پیش‌پردازش و استانداردسازی ویژگی‌های مربوط به دسته‌بندی، موقعیت و اطلاعات عمومی آگهی.
    """
    df = df.copy()

    # ستون‌های متنی و دسته‌بندی
    cols = [
        'cat2_slug', 'cat3_slug', 'city_slug', 'neighborhood_slug', 
        'user_type', 'deed_type', 'building_direction', 'floor_material'
    ]
    
    # ۱. حذف ستون با درصد بالای مفقودی در صورت وجود
    if 'property_type' in df.columns:
        df = df.drop(columns=['property_type'])

    # ۲. استانداردسازی حروف فارسی و فاصله‌های اضافه برای تمامی ستون‌های متنی
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string") 
                .str.strip()
                .str.replace('ي', 'ی', regex=False)
                .str.replace('ك', 'ک', regex=False)
                .str.replace('\u200c', ' ', regex=False)
            )

    # ۳. پر کردن مقادیر NaN و جایگزینی unselect با unknown
    cols_to_fill = [
        'neighborhood_slug', 'user_type', 'deed_type', 
        'building_direction', 'floor_material'
    ]
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].replace(['unselect', 'nan', 'None'], np.nan).fillna('unknown')

    # ۴. حذف سطرهایی که دسته‌بندی یا شهر مشخصی ندارند
    df = df.dropna(subset=['cat3_slug', 'city_slug']).reset_index(drop=True)

    return df


def preprocess_numerical_building_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    پیش‌پردازش و استانداردسازی ویژگی‌های عددی مرتبط با مشخصات ساختمان و املاک.
    """
    df = df.copy()

    # ۱. استانداردسازی سال ساخت (تبدیل اعداد فارسی، استخراج عدد و اعتبارسنجی بازه)
    if 'construction_year' in df.columns:
        trans_table = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')
        df['construction_year'] = (
            df['construction_year']
            .astype(str)
            .str.translate(trans_table)
            .str.extract(r'(\d+)')[0]
        )
        df['construction_year'] = pd.to_numeric(df['construction_year'], errors='coerce').astype('Int64')

        # اعتبارسنجی بازه‌ی منطقی سال ساخت (شمسی): بین ۱۲۹۰ تا سال جاری
        current_year = 1404
        invalid_year_mask = (df['construction_year'] < 1290) | (df['construction_year'] > current_year)
        df.loc[invalid_year_mask, 'construction_year'] = pd.NA

    # ۲. نگاشت تعداد اتاق‌ها (rooms_count)
    if 'rooms_count' in df.columns:
        rooms_map = {
            'بدون اتاق': 0,
            'یک': 1,
            'دو': 2,
            'سه': 3,
            'چهار': 4,
            'پنج یا بیشتر': 5
        }
        df['rooms_count'] = df['rooms_count'].map(rooms_map).astype('Int64')

    # ۳. پاک‌سازی ستون طبقه (floor)
    if 'floor' in df.columns:
        df['floor'] = (
            df['floor']
            .astype(str)
            .str.strip()
            .str.replace('30+', '30', regex=False)
            .str.replace(r'\.0$', '', regex=True)
        )
        df['floor'] = pd.to_numeric(df['floor'], errors='coerce').astype('Int64')

    # ۴. پاک‌سازی کل طبقات (total_floors_count)
    if 'total_floors_count' in df.columns:
        df['total_floors_count'] = (
            df['total_floors_count']
            .astype(str)
            .str.strip()
            .str.replace('unselect', '', regex=False)
            .str.replace('30+', '30', regex=False)
            .str.replace(r'\.0$', '', regex=True)
        )
        df['total_floors_count'] = pd.to_numeric(df['total_floors_count'], errors='coerce').astype('Int64')

    # ۵. پاک‌سازی تعداد واحد در طبقه (unit_per_floor)
    if 'unit_per_floor' in df.columns:
        df['unit_per_floor'] = (
            df['unit_per_floor']
            .astype(str)
            .str.strip()
            .str.replace('more_than_8', '9', regex=False)
            .str.replace('unselect', '', regex=False)
        )
        df['unit_per_floor'] = pd.to_numeric(df['unit_per_floor'], errors='coerce').astype('Int64')

    # ۵.۵. اعتبارسنجی منطقی: طبقه نباید از کل طبقات بیشتر باشد
    if 'floor' in df.columns and 'total_floors_count' in df.columns:
        invalid_floor_mask = (
            df['floor'].notna() 
            & df['total_floors_count'].notna() 
            & (df['floor'] > df['total_floors_count'])
        )
        df.loc[invalid_floor_mask, ['floor', 'total_floors_count']] = pd.NA

    # ۶. ساخت ویژگی باینری has_land و حذف land_size
    if 'land_size' in df.columns:
        df['has_land'] = df['land_size'].notnull().astype('Int64')
        df = df.drop(columns=['land_size'])

    # ۷. مدیریت داده‌های پرت متراژ (building_size)
    if 'building_size' in df.columns and 'cat2_slug' in df.columns:
        df['building_size'] = pd.to_numeric(df['building_size'], errors='coerce')
        
        # املاک مسکونی
        residential_mask = df['cat2_slug'].str.contains('residential', na=False)
        df.loc[residential_mask & ((df['building_size'] < 10) | (df['building_size'] > 1000)), 'building_size'] = np.nan

        # املاک تجاری/اداری
        commercial_mask = df['cat2_slug'].str.contains('commercial', na=False)
        df.loc[commercial_mask & ((df['building_size'] < 10) | (df['building_size'] > 20000)), 'building_size'] = np.nan

        # سایر دسته‌ها
        other_mask = ~residential_mask & ~commercial_mask
        df.loc[other_mask & ((df['building_size'] < 10) | (df['building_size'] > 10000)), 'building_size'] = np.nan

    return df


def preprocess_daily_rental_and_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    پیش‌پردازش و تمیزکاری ویژگی‌های مربوط به اجاره روزانه و متون (عنوان و توضیحات آگهی).
    """
    out = df.copy()

    # ۱. پیش‌پردازش ظرفیت‌های عددی املاک اجاره روزانه
    if "regular_person_capacity" in out.columns:
        out["regular_person_capacity"] = pd.to_numeric(
            out["regular_person_capacity"], errors="coerce"
        ).astype("Int64")

    if "extra_person_capacity" in out.columns:
        out["extra_person_capacity"] = (
            out["extra_person_capacity"]
            .astype("string")
            .str.strip()
            .replace("30+", "30")
            .pipe(pd.to_numeric, errors="coerce")
            .astype("Int64")
        )

    # ۲. پاک‌سازی و استانداردسازی قیمت‌های اجاره روزانه
    price_cols = [
        "cost_per_extra_person",
        "rent_price_on_regular_days",
        "rent_price_on_special_days",
        "rent_price_at_weekends",
    ]

    extreme_price_threshold = 1_000_000_000

    for col in price_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[out[col] > extreme_price_threshold, col] = np.nan
            out[f"{col}_suspicious_low"] = out[col].notna() & (out[col] <= 10_000)

    # ۳. نرم‌سازی متون (عنوان و توضیحات)
    text_cols = ["title", "description"]

    for col in text_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype("string")
                .str.replace("ي", "ی", regex=False)
                .str.replace("ك", "ک", regex=False)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .replace("", pd.NA)
            )

    # ۴. استخراج ویژگی‌های سبک متنی (طول کاراکتر و تعداد کلمات)
    if "title" in out.columns:
        out["title_length"] = out["title"].str.len()
        out["title_word_count"] = out["title"].str.split().str.len().astype("Int64")   # ← astype اضافه شد

    if "description" in out.columns:
        out["description_length"] = out["description"].str.len()
        out["description_word_count"] = out["description"].str.split().str.len().astype("Int64")   # ← astype اضافه شد
    return out


def preprocess_amenity_and_facility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    پیش‌پردازش و تمیزکاری امکانات اصلی ملک (ویژگی‌های بولین) و سیستم‌های تأسیساتی.
    """
    out = df.copy()

    # ۱. حذف ستون بدون پوشش داده (نال تقریباً ۱۰۰٪)
    drop_bool_cols = ['rent_to_single']
    out = out.drop(columns=drop_bool_cols, errors='ignore')

    # ۲. استانداردسازی همه‌ی ویژگی‌های بولین و تبدیل به boolean
    #    (شامل ستون‌های کم‌پوشش که قبلاً حذف می‌شدند - طبق پیشنهاد دوستمون
    #    اینها نگهداری می‌شوند تا برای تحلیل توزیع مکانی امکاناتی مثل
    #    استخر/باربیکیو/نگهبان قابل استفاده باشند)
    main_bool_cols = [
        'has_balcony', 'has_elevator', 'has_warehouse', 'has_parking', 'is_rebuilt',
        'has_business_deed', 'has_water', 'has_electricity', 'has_gas',
        'has_security_guard', 'has_barbecue', 'has_pool', 'has_jacuzzi', 'has_sauna'
    ]
    
    bool_map = {
        'true': True,
        'false': False,
        'unselect': np.nan,
        'nan': np.nan,
        'none': np.nan
    }

    for col in main_bool_cols:
        if col in out.columns:
            cleaned_series = out[col].astype(str).str.lower().str.strip().map(bool_map)
            out[col] = cleaned_series.astype('boolean')

    # ۳. استانداردسازی ویژگی‌های دسته‌ای تأسیسات (سرمایش، گرمایش، سرویس بهداشتی و ...)
    facility_cols = [
        'has_warm_water_provider', 
        'has_heating_system', 
        'has_cooling_system', 
        'has_restroom'
    ]

    for col in facility_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype("string")   
                .str.lower()
                .str.strip()
                .replace({'unselect': pd.NA, 'nan': pd.NA, 'none': pd.NA})  
            )
    return out


def preprocess_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    پیش‌پردازش و پاک‌سازی ویژگی‌های مالی (رهن، اجاره، قیمت فروش و قابلیت‌های تبدیل مالی).
    """
    out = df.copy()

    mode_type_cols = ['price_mode', 'credit_mode', 'rent_mode', 'rent_type']  # ← بلوک جدید
    for col in mode_type_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype("string")
                .replace(['unselect', 'nan', 'None'], pd.NA)
            )
        
    # تبدیل ستون‌های مالی به عددی جهت اطمینان
    num_fin_cols = ["rent_value", "credit_value", "price_value",
                "transformable_credit", "transformed_credit",
                "transformable_rent", "transformed_rent"]
    
    for col in num_fin_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # ۱. تبدیل و اصلاح نوع ستون‌های بولین تبدیل قیمت/رهن
    bool_transform_cols = ["rent_credit_transform", "transformable_price"]
    
    for col in bool_transform_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .replace({"True": True, "False": False})
                .astype("boolean")
            )

    transformed_cols = ["transformed_credit", "transformed_rent"]
    suspicious_low_threshold = 10_000

    for col in transformed_cols:
        if col in out.columns:
            out[f"{col}_suspicious_low"] = out[col].notna() & (out[col] <= suspicious_low_threshold)

    # ۲. استخراج ویژگی جدید برای تشخیص هوشمند «رهن کامل» (inferred_full_credit)
    # ۲. استخراج ویژگی جدید برای تشخیص هوشمند «رهن کامل» (inferred_full_credit)
    out["inferred_full_credit"] = pd.Series(pd.NA, index=out.index, dtype="boolean")

    if "cat2_slug" in out.columns:
        rental_category = out["cat2_slug"].str.contains("rent", case=False, na=False)
        explicit_full_credit = out.get("rent_type") == "full_credit"

        # شرایط رهن کامل قوی (اجاره صفر، مبلغ رهن مثبت، بدون قیمت فروش)
        strong_full_credit = (
            rental_category
            & (out.get("rent_value") == 0)
            & (out.get("credit_value") > 0)
            & out.get("price_mode").isna()
            & out.get("price_value").isna()
        )

        out.loc[explicit_full_credit | strong_full_credit, "inferred_full_credit"] = True

        # املاکی که قطعا رهن کامل نیستند
        definitely_not_full_credit = (
            (out.get("rent_value") > 0)
            | out.get("price_value").notna()
            | out.get("price_mode").notna()
        )

        out.loc[
            definitely_not_full_credit & ~explicit_full_credit,
            "inferred_full_credit"
        ] = False

    return out


def preprocess_time_and_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    پیش‌پردازش و تمیزکاری جامع ویژگی‌های زمانی (شمسی/میلادی) و مکان‌محور (مختصات، اعتبار پین و فاصله تا مرکز شهر).
    """
    out = df.copy()

    # ۱. حذف ستون‌های کم‌کاربرد
    drop_cols = ["location_radius", "rent_to_single"]
    out = out.drop(columns=drop_cols, errors="ignore")

    # ۲. پیش‌پردازش زمان و استخراج تقویم شمسی
    if "created_at_month" in out.columns:
        created_dt = pd.to_datetime(out["created_at_month"], errors="coerce")
        
        # استخراج ویژگی‌های میلادی پایه
        out["created_year"] = created_dt.dt.year.astype("Int64")
        out["created_month"] = created_dt.dt.month.astype("Int64")
        
        # محاسبه month_index و is_main_period (دوره اصلی آگهی‌ها)
        first_date = created_dt.min()
        if pd.notna(first_date):
            out["month_index"] = (
                (created_dt.dt.year - first_date.year) * 12 + (created_dt.dt.month - first_date.month)
            ).astype("Int64")
        
        out["is_main_period"] = created_dt.between("2024-05-01", "2024-12-01").astype("boolean")

        # تبدیل به شمسی با jdatetime
        try:
            import jdatetime
            
            month_names = [
                "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
                "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"
            ]
            
            unique_months = created_dt.dropna().unique()
            to_jalali = {}
            for m in unique_months:
                mid_month = (m + pd.Timedelta(days=14)).date()
                to_jalali[m] = jdatetime.date.fromgregorian(date=mid_month)

            jalali_series = created_dt.map(to_jalali)
            
            out["jalali_year"] = jalali_series.map(lambda d: d.year if pd.notna(d) else np.nan).astype("Int64")
            out["jalali_month"] = jalali_series.map(lambda d: d.month if pd.notna(d) else np.nan).astype("Int64")
            
            def make_jalali_label(d):
                if pd.isna(d):
                    return pd.NA
                return f"{d.year} {month_names[d.month - 1]}"

            out["jalali_label"] = jalali_series.map(make_jalali_label).astype("string")

        except ImportError:
            print("کتابخانه 'jdatetime' نصب نیست. استخراج تقویم شمسی نادیده گرفته شد.")

    # ۳. اعتبارسنجی و پاک‌سازی مختصات جغرافیایی
    lat_col, lon_col = "location_latitude", "location_longitude"
    
    if lat_col in out.columns and lon_col in out.columns:
        out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
        out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
        
        out["coord_status"] = "ok"
        out.loc[out[lat_col].isna() | out[lon_col].isna(), "coord_status"] = "missing"

        # الف) فیلتر محدوده جغرافیایی ایران
        inside_iran = out[lat_col].between(25.0, 40.0) & out[lon_col].between(44.0, 64.0)
        outside = out[lat_col].notna() & ~inside_iran
        out.loc[outside, "coord_status"] = "kharej"
        out.loc[outside, [lat_col, lon_col]] = np.nan

        # ب) شناسایی پین‌های اشتباه (پین‌های یکسان ثبت‌شده در چند شهر مختلف)
        has_coord = out[lat_col].notna() & out[lon_col].notna()
        if has_coord.any() and "city_slug" in out.columns:
            pins = out[has_coord].groupby([lat_col, lon_col])
            out["pin_repeat_count"] = pins[lat_col].transform("size").astype("Int64")
            
            cities_per_pin = pins["city_slug"].transform("nunique")
            bad_idx = cities_per_pin[cities_per_pin > 1].index
            
            out.loc[bad_idx, "coord_status"] = "pin_chand_shahri"
            out.loc[bad_idx, [lat_col, lon_col]] = np.nan

        # ج) محاسبه فاصله تا مرکز شهر (Haversine) و حذف نقاط پرت (> ۵۰ کیلومتر)
        has_valid_coord = out[lat_col].notna() & out[lon_col].notna()
        if has_valid_coord.any() and "city_slug" in out.columns:
            city_counts = out.groupby("city_slug")[lat_col].count()
            centers = out.groupby("city_slug")[[lat_col, lon_col]].median()
            valid_centers = centers[city_counts >= 30]

            center_lat = out["city_slug"].map(valid_centers[lat_col])
            center_lon = out["city_slug"].map(valid_centers[lon_col])

            # فرمول Haversine به کیلومتر
            lat1, lon1 = np.radians(out[lat_col]), np.radians(out[lon_col])
            lat2, lon2 = np.radians(center_lat), np.radians(center_lon)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
            out["dist_to_city_km"] = (6371 * 2 * np.arcsin(np.sqrt(a))).round(1)

            far_mask = out["dist_to_city_km"] > 50
            out.loc[far_mask, "coord_status"] = "door_az_shahr"
            out.loc[far_mask, [lat_col, lon_col]] = np.nan

        # د) تبدیل به UTM (زون 39)
        try:
            import utm

            valid_mask = out[lat_col].notna() & out[lon_col].notna()
            out["location_utm_x"] = np.nan
            out["location_utm_y"] = np.nan

            if valid_mask.any():
                lats = out.loc[valid_mask, lat_col].values
                lons = out.loc[valid_mask, lon_col].values
                utm_x, utm_y, _, _ = utm.from_latlon(lats, lons, force_zone_number=39)
                out.loc[valid_mask, "location_utm_x"] = utm_x
                out.loc[valid_mask, "location_utm_y"] = utm_y

        except ImportError:
            print("کتابخانه 'utm' نصب نیست. محاسبه ستون‌های location_utm نادیده گرفته شد.")
        out = out.drop(columns=["created_at_month"])   # ← این خط اضافه بشه

    return out


def extract_amenities_from_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    پر کردن مقادیر نال در ۵ ستون اصلی امکانات از طریق جستجو در متن (بدون ایجاد ستون جدید).
    """
    out = df.copy()

    # ۱. یکسان‌سازی متنی عنوان و توضیحات
    text_corpus = (
        (out['title'].fillna('') + ' ' + out['description'].fillna(''))
        .astype(str)
        .str.replace('ي', 'ی', regex=False)
        .str.replace('ك', 'ک', regex=False)
        .str.lower()
    )

    # ۲. الگوهای جامع (Regex) برای پوشش نگارش‌های مختلف
    patterns = {
        'has_balcony': 'بالکن|تراس|بهارخواب|پاسیو',
        'has_elevator': 'آسانسور|اسانسور|الواتور',
        'has_security_guard': 'نگهبان|لابی‌من|لابی من|سرایدار|حراست',
        'has_barbecue': 'باربیکیو|باربکیو|باربیکو|کباب‌پز|کباب پز',
        'has_pool': 'استخر',
    }

    # ۳. بازنویسی و اصلاح مستقیم همان ستون‌های اصلی
    for col, pattern in patterns.items():
        text_match = text_corpus.str.contains(pattern, regex=True)
        if col in out.columns:
            # ترکیب ارزش قبلی با متن و بازنویسی روی همان ستون
            existing_val = out[col] == True
            out[col] = (existing_val | text_match).astype('boolean')

    return out

def run_full_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    اجرای کامل و متوالی تمامی توابع پیش‌پردازش رو مجموعه داده.
    """
    out = df.copy()
    out = out.drop_duplicates()  
    out = preprocess_category_location_info(out)
    out = preprocess_numerical_building_features(out)
    out = preprocess_daily_rental_and_text_features(out)
    out = preprocess_amenity_and_facility_features(out)
    out = extract_amenities_from_text(out)
    out = preprocess_financial_features(out)
    out = preprocess_time_and_location_features(out)
    return out