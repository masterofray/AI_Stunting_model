# Analisis Statistik dan Data Science Data Stunting Kabupaten Mamberamo Raya

**Dataset:** `DataProc.csv`  
**Tanggal analisis:** 16 September 2026  
**Observasi:** 607 baris  
**Variabel:** 9 kolom

## 1. Executive summary

Dataset berisi 607 observasi dan 9 variabel: distrik, status stunting, gender, `Malnutrition_Level`, serta lima variabel biner terkait kondisi/paparan. Tidak terdapat missing value pada sembilan variabel tersebut.

Temuan paling penting justru berada pada kualitas unit observasi. Ada **279 baris duplikat persis di luar kemunculan pertama**, sehingga hanya terdapat **328 kombinasi record yang unik secara persis**. Terdapat 125 kelompok record berulang, 404 baris berada di dalam kelompok berulang, dan satu record muncul 24 kali. Karena dataset tidak mempunyai `Child_ID`, duplikasi tersebut **tidak dapat dipastikan sebagai duplikasi individu**. Bisa saja sebagian merupakan anak berbeda yang mempunyai seluruh karakteristik tercatat sama. Karena itu, penghapusan duplikat dijadikan **analisis sensitivitas**, bukan keputusan cleaning final.

Prevalensi `Stunting=1` pada file mentah adalah **54.0% (328/607)** dengan interval 95% Wilson sekitar **50.1%–58.0%**. Pada data exact-deduplicated, prevalensi menjadi **51.8% (170/328)** dengan interval sekitar **46.4%–57.2%**. Perbedaan ini cukup besar untuk membuat inferensi tanpa audit ID menjadi berisiko.

Terdapat heterogenitas yang kuat antar-distrik. Pada file mentah, prevalensi teramati berkisar dari **13.3% di Mamberamo Tengah** hingga **80.7% di Waropen Atas**. Uji asosiasi distrik dengan stunting memberikan **Cramer's V=0.576, p<0.001**, menunjukkan hubungan yang besar pada level sampel ini. Setelah exact deduplication, rentangnya menjadi **18.8%–71.6%** dan Cramer's V turun menjadi **0.410**, tetapi heterogenitas distrik tetap terlihat.

Beberapa variabel biner menunjukkan asosiasi univariat yang kuat dengan `Stunting`: `Exclusive_Milky` dan `Difficult_Acess` terutama menonjol. Namun, **asosiasi ini tidak boleh langsung dibaca sebagai efek kausal**, karena tidak tersedia informasi urutan waktu, definisi operasional yang lengkap, sampling design, dan beberapa variabel dapat merupakan indikator yang saling tumpang tindih dengan konstruk stunting.

`Malnutrition_Level` memerlukan perhatian khusus. Pada data ini, kode **2–6 seluruhnya memiliki `Stunting=1`**, sedangkan kode 0 dan 1 memiliki sebagian `Stunting=0` dan sebagian `Stunting=1`. Secara statistik hal ini menciptakan pemisahan sempurna/nyaris sempurna pada regresi. Lebih penting lagi, kita belum memiliki **codebook** yang menjelaskan arti kode 0–6. Variabel tersebut tidak dimasukkan ke Bayesian mixed-effects logistic regression utama agar tidak menghasilkan inferensi palsu dari variabel yang sangat mungkin merupakan label yang tumpang tindih dengan outcome.

Analisis Bayesian menghasilkan dua lapisan: (1) partial pooling prevalence distrik menggunakan Beta-Binomial empiris-Bayes dan (2) Bayesian hierarchical logistic regression dengan random intercept distrik. Untuk probabilitas stunting, random-effect SD berada sekitar 0.179 pada raw data dan 0.203 setelah exact deduplication; logistic ICC kira-kira **0.010–0.012**. Artinya, setelah kovariat dimasukkan, sebagian besar variasi pada model ini lebih banyak ditangkap oleh kovariat/fixed effects daripada random intercept distrik, walaupun perbedaan distrik tetap terlihat pada fixed effects dan prevalence mentah.

Model machine-learning untuk prediksi menghasilkan AUC cross-validation **0.921 pada raw data**, tetapi turun menjadi **0.858 setelah exact deduplication**. Penurunan ini merupakan sinyal penting bahwa duplikasi dapat memperbesar performa validasi melalui kemiripan/leakage. Selain itu, `Malnutrition_Level` sangat dekat dengan outcome, sehingga performa tinggi tidak boleh dianggap sebagai bukti kemampuan prediksi klinis.

---

## 2. Struktur dan kualitas data

| Aspek | Hasil |
|---|---:|
| Jumlah baris | 607 |
| Jumlah kolom | 9 |
| Missing value | 0 |
| Distrik unik | 7 |
| Exact-unique records | 328 |
| Exact duplicate rows beyond first | 279 |
| Duplicate groups | 125 |
| Rows inside duplicate groups | 404 |
| Maximum repetition of one exact record | 24 |
| `Stunting=1` | 328 (54.0%) |

Kolom yang tersedia:

`District_Name`, `Stunting`, `Gender`, `Malnutrition_Level`, `Exclusive_Milky`, `Smoke_Habit`, `PureWater_Access`, `Healthy_Toilet`, `Difficult_Acess`.

Kolom yang dibutuhkan untuk sebagian analisis yang diminta tetapi **tidak ada** di file:

`Enumerator`, `Puskesmas`, `Perawat`, tanggal/waktu pengukuran, `Child_ID`, umur/umur-bulan, tinggi badan, berat badan, HAZ/WAZ/WHZ atau z-score antropometri.

### Implikasi

Analisis perawat, puskesmas, enumerator, temporal, dan pemeriksaan antropometri **tidak dapat dilakukan secara empiris dari file ini**. Tidak ada dasar statistik yang sah untuk mengisi variabel tersebut dari asumsi.

Untuk versi dataset berikutnya, minimal diperlukan identifier dan metadata:

- `Child_ID`
- `Enumerator_ID`
- `Puskesmas_ID`
- `Perawat_ID` atau petugas pengukuran
- `Survey_Date`
- `Age_Months`
- `Sex`
- `Weight_kg`
- `Height_cm`
- z-score WHO yang digunakan dan/atau variabel dasar untuk menghitungnya
- flag kualitas pengukuran dan alasan missing/outlier

---

## 3. Prevalensi stunting keseluruhan

Pada 607 baris mentah, `Stunting=1` tercatat pada 328 observasi sehingga prevalence sampel adalah **54.0%**. Wilson 95% CI sekitar **50.1%–58.0%**.

Setelah hanya menghapus record yang identik secara penuh, tersisa 328 record dan prevalence menjadi **51.8%**, Wilson 95% CI sekitar **46.4%–57.2%**.

Perbedaan antara 54.0% dan 51.8% menunjukkan bahwa keputusan tentang unit analisis/duplikasi tidak sekadar masalah teknis kecil.

---

## 4. Analisis per distrik

### 4.1 Raw data

| Distrik | n | Stunting n | Prevalensi | 95% CI Wilson |
|---|---:|---:|---:|---:|
| BENUKI | 51 | 33 | 64.7% | 51.0%–76.4% |
| MAMBERAMO HILIR | 65 | 20 | 30.8% | 20.9%–42.8% |
| MAMBERAMO HULU | 35 | 7 | 20.0% | 10.0%–35.9% |
| MAMBERAMO TENGAH | 120 | 16 | 13.3% | 8.4%–20.6% |
| MAMBERAMO TENGAH TIMUR | 19 | 3 | 15.8% | 5.5%–37.6% |
| SAWAI | 141 | 107 | 75.9% | 68.2%–82.2% |
| WAROPEN ATAS | 176 | 142 | 80.7% | 74.2%–85.8% |

Uji chi-square distrik vs stunting memberikan **χ²=201.48, df=6, p≈9.19×10⁻⁴¹**, dengan **Cramer's V=0.576**.

Angka ini menggambarkan asosiasi kuat dalam dataset, bukan bukti bahwa lokasi distrik secara kausal menyebabkan stunting. Perbedaan dapat mencerminkan komposisi sampel, kondisi sosial-lingkungan, pola pengukuran, distribusi petugas, atau faktor lain yang tidak tersedia.

### 4.2 Sensitivitas exact-deduplicated

| Distrik | n unik | Stunting n | Prevalensi |
|---|---:|---:|---:|
| BENUKI | 36 | 21 | 58.3% |
| MAMBERAMO HILIR | 45 | 16 | 35.6% |
| MAMBERAMO HULU | 24 | 6 | 25.0% |
| MAMBERAMO TENGAH | 52 | 14 | 26.9% |
| MAMBERAMO TENGAH TIMUR | 16 | 3 | 18.8% |
| SAWAI | 88 | 63 | 71.6% |
| WAROPEN ATAS | 67 | 47 | 70.1% |

Setelah exact deduplication, chi-square distrik vs stunting menjadi **χ²=55.01, df=6, p≈4.62×10⁻¹⁰**, Cramer's V **0.410**. Heterogenitas masih ada, tetapi kekuatan asosiasinya mengecil.

---

## 5. Duplicate detection dan implikasi statistik

`DataProc.csv` tidak mempunyai `Child_ID`, sehingga definisi “duplicate” yang dapat diuji saat ini hanyalah **duplikasi seluruh baris**.

Temuan:

- 279 duplicate rows beyond first.
- 125 fingerprint groups memiliki pengulangan >1.
- 404 dari 607 baris berada dalam kelompok yang mempunyai setidaknya satu salinan lain.
- Satu kombinasi record muncul 24 kali.

Contoh kelompok berulang terbesar berada di **WAROPEN ATAS** dan memiliki status `Stunting=1`, `Malnutrition_Level=2`, `Exclusive_Milky=1`, dan `Difficult_Acess=1`, antara lain muncul 24 kali dan 22 kali.

### Mengapa tidak boleh langsung menghapus semua duplikat?

Karena dua anak yang berbeda dapat secara sah mempunyai kombinasi kategori yang sama persis. Tanpa ID individu, kita tidak tahu apakah 24 baris identik adalah satu anak tercatat 24 kali atau 24 anak yang kebetulan mempunyai profil kategori identik.

### Sensitivity analysis

Perubahan estimasi odds ratio dari raw ke exact-deduplicated cukup besar:

| Variabel | OR raw | OR dedup | Fisher p raw | Fisher p dedup |
|---|---:|---:|---:|---:|
| Gender | 1.18 | 1.30 | 0.365 | 0.269 |
| Exclusive_Milky | 6.24 | 2.85 | <0.001 | <0.001 |
| Smoke_Habit | 1.22 | 1.43 | 0.223 | 0.121 |
| PureWater_Access | 0.71 | 1.07 | 0.034 | 0.825 |
| Healthy_Toilet | 0.43 | 0.63 | <0.001 | 0.054 |
| Difficult_Acess | 13.12 | 4.96 | <0.001 | <0.001 |

Perhatikan `PureWater_Access`: arah asosiasi berubah dari OR <1 menjadi sekitar OR=1 setelah dedup. Ini contoh konkret mengapa inferensi harus menunggu rekonstruksi unit observasi yang benar.

---

## 6. Analisis bivariat faktor/karakteristik

### Exclusive_Milky

Pada raw data, prevalensi stunting:

- kategori 1: 74.1%
- kategori 0: 31.5%
- RR ≈ 2.36
- OR ≈ 6.24
- Fisher exact p ≈ 2.53×10⁻²⁶

Asosiasi ini sangat kuat pada dataset, tetapi arti substantif dari `Exclusive_Milky=1` harus dikonfirmasi lewat codebook. Nama kolom sendiri belum menjelaskan apakah 1 berarti “mendapat ASI eksklusif”, “tidak mendapat”, atau definisi lain.

### Smoke_Habit

OR raw sekitar 1.22 dengan Fisher p≈0.223. Pada exact-deduplicated data OR sekitar 1.43 dengan p≈0.121. Bukti univariat pada dataset ini tidak stabil/kuat untuk menyimpulkan adanya asosiasi non-zero.

### PureWater_Access

OR raw sekitar 0.71 dengan p≈0.034, namun menjadi OR≈1.07 dan p≈0.825 setelah exact deduplication. Temuan ini sangat sensitif terhadap duplikasi.

### Healthy_Toilet

OR raw sekitar 0.43 dengan p≈1.5×10⁻⁶, sedangkan setelah exact deduplication OR sekitar 0.63 dengan p≈0.054. Lagi-lagi, data quality memengaruhi interpretasi.

### Difficult_Acess

OR raw sekitar 13.12 dengan p≈3.28×10⁻³⁶; setelah exact deduplication OR sekitar 4.96 dengan p≈9.90×10⁻¹⁰. Arah dan adanya asosiasi bertahan, tetapi besarnya efek turun cukup banyak.

### Gender

OR raw sekitar 1.18 dengan p≈0.365 dan OR dedup sekitar 1.30 dengan p≈0.269. Tidak ada bukti univariat yang kuat pada dataset ini.

---

## 7. `Malnutrition_Level`: temuan struktural yang sangat penting

Distribusi `Malnutrition_Level` terhadap stunting adalah:

| Kode | Non-stunting n | Stunting n | Prevalensi stunting |
|---:|---:|---:|---:|
| 0 | 106 | 18 | 14.5% |
| 1 | 173 | 54 | 23.8% |
| 2 | 0 | 176 | 100.0% |
| 3 | 0 | 38 | 100.0% |
| 4 | 0 | 27 | 100.0% |
| 5 | 0 | 11 | 100.0% |
| 6 | 0 | 4 | 100.0% |

Uji chi-square memberikan **χ²=379.35, df=6, p≈7.66×10⁻⁷⁹**.

Namun angka tersebut bukan berarti “malnutrition level menyebabkan stunting”. Justru struktur yang terlalu deterministik mengindikasikan salah satu kemungkinan berikut:

1. `Malnutrition_Level` adalah label yang secara definisi diturunkan dari status stunting atau sangat terkait dengannya.
2. Kode 2–6 merupakan kategori status gizi yang memang hampir identik dengan outcome yang dilabelkan sebagai stunting.
3. Ada proses encoding/ETL yang salah.
4. Ada variable derivation yang dilakukan menggunakan target, sehingga terjadi target leakage.

Karena arti 0–6 belum tersedia, **jangan mengonversi kode tersebut menjadi ordinal “lebih tinggi = lebih buruk” tanpa codebook**.

Dalam Bayesian logistic utama, variabel ini sengaja dikeluarkan untuk mencegah hasil yang tampak hebat secara statistik tetapi tidak valid secara inferensial.

---

## 8. Bayesian district prevalence dengan partial pooling

Model yang digunakan adalah Beta-Binomial empiris-Bayes:

\[
 y_d \sim \text{Binomial}(n_d, p_d),
\qquad
 p_d \sim \text{Beta}(\alpha,\beta)
\]

Hyperprior diestimasi melalui maximum marginal likelihood pada tujuh distrik.

Estimasi prior bersama:

- α ≈ 1.377
- β ≈ 1.752

Posterior mean per distrik:

| Distrik | Raw prevalence | Posterior mean | 95% Bayesian CI |
|---|---:|---:|---:|
| BENUKI | 64.7% | 63.5% | 50.4%–75.7% |
| MAMBERAMO HILIR | 30.8% | 31.4% | 21.0%–42.8% |
| MAMBERAMO HULU | 20.0% | 22.0% | 10.5%–36.2% |
| MAMBERAMO TENGAH | 13.3% | 14.1% | 8.6%–20.8% |
| MAMBERAMO TENGAH TIMUR | 15.8% | 19.8% | 6.4%–38.3% |
| SAWAI | 75.9% | 75.2% | 67.9%–81.9% |
| WAROPEN ATAS | 80.7% | 80.0% | 73.9%–85.5% |

Partial pooling menarik estimasi distrik dengan sample kecil sedikit menuju pola keseluruhan. Ini sangat berguna ketika ukuran sampel antar-distrik tidak seimbang, seperti 19 observasi di Mamberamo Tengah Timur versus 176 di Waropen Atas.

---

## 9. Bayesian hierarchical logistic regression

Model:

\[
\text{logit}(P(Y_i=1)) = \beta_0 + X_i\beta + \gamma_{district[i]}
\]

Dengan random intercept distrik:

\[
\gamma_d \sim N(0, \sigma_{district}^2)
\]

`Malnutrition_Level` dikeluarkan dari model utama karena pemisahan hampir/sepenuhnya sempurna dan risiko label leakage.

### Raw data: posterior odds ratio

| Parameter | OR | Approx. 95% Bayesian interval |
|---|---:|---:|
| Gender | 1.49 | 1.13–1.95 |
| Exclusive_Milky | 2.75 | 2.08–3.64 |
| Smoke_Habit | 1.64 | 1.23–2.18 |
| PureWater_Access | 1.18 | 0.88–1.58 |
| Healthy_Toilet | 1.10 | 0.77–1.58 |
| Difficult_Acess | 2.93 | 2.33–3.69 |
| Sawai vs Benuki | 1.75 | 1.17–2.61 |
| Waropen Atas vs Benuki | 2.96 | 2.02–4.34 |

### Exact-deduplicated data: posterior odds ratio

| Parameter | OR | Approx. 95% Bayesian interval |
|---|---:|---:|
| Gender | 1.42 | 1.01–1.99 |
| Exclusive_Milky | 1.92 | 1.36–2.71 |
| Smoke_Habit | 1.55 | 1.11–2.18 |
| PureWater_Access | 1.53 | 1.09–2.16 |
| Healthy_Toilet | 1.07 | 0.72–1.59 |
| Difficult_Acess | 2.15 | 1.62–2.85 |
| Sawai vs Benuki | 1.92 | 1.19–3.09 |
| Waropen Atas vs Benuki | 2.10 | 1.23–3.59 |

Perubahan parameter antara raw dan dedup menunjukkan **duplicate structure materially changes multivariable inference**.

Random-effect estimates:

- Raw: district SD ≈ 0.179; logistic ICC ≈ 0.010.
- Dedup: district SD ≈ 0.203; logistic ICC ≈ 0.012.

ICC kecil bukan berarti distrik “tidak penting”. Ia berarti bahwa setelah kovariat dalam model dimasukkan, random intercept distrik menjelaskan proporsi kecil dari latent logistic variance model tersebut. Prevalence mentah antar-distrik tetap sangat berbeda.

---

## 10. Machine learning / predictive validation

Sebagai diagnostic data-science layer, digunakan logistic regression dengan one-hot encoding untuk distrik dan `Malnutrition_Level`, lalu 5-fold stratified cross-validation.

| Dataset | AUC | Average Precision | Brier |
|---|---:|---:|---:|
| Raw, 607 rows | 0.921 | 0.953 | 0.094 |
| Exact-deduplicated, 328 rows | 0.858 | 0.907 | — |

Kenaikan AUC pada raw data tidak boleh dianggap sebagai keberhasilan model deployment. Ada dua alasan penting:

1. Duplikasi dapat membuat fold train dan test memiliki record identik/nyaris identik.
2. `Malnutrition_Level` mempunyai hubungan ekstrem dengan outcome dan dapat bertindak sebagai target leakage.

Kesimpulan yang sah dari analisis ini adalah bahwa **data structure sangat mudah diprediksi**, bukan bahwa kita sudah mempunyai model screening klinis yang valid.

---

## 11. Analisis yang diminta tetapi belum bisa dijalankan

### Perawat / Puskesmas / Enumerator

Tidak ada identifier `Perawat`, `Puskesmas`, atau `Enumerator`. Karena itu tidak mungkin menghitung:

- prevalence per petugas;
- volume entry per petugas;
- duplicate rate per enumerator;
- missingness rate per petugas;
- inter-enumerator variation;
- measurement bias menurut petugas;
- clustering by puskesmas/enumerator;
- reliability metrics atau inter-rater agreement.

Begitu identifier tersebut tersedia, model yang disarankan adalah hierarchical model berlapis, misalnya:

\[
logit(p_i)=\beta X_i+u_{district}+v_{puskesmas}+w_{enumerator}
\]

Jika anak diukur lebih dari sekali, repeated-measures structure juga diperlukan.

### Analisis temporal

Tidak ada tanggal atau timestamp. Karena itu tidak valid untuk menghitung:

- tren bulanan/kuartalan;
- seasonality;
- changepoint;
- sebelum/sesudah intervensi;
- EWMA/CUSUM untuk surveillance;
- lagged exposure effects.

Dataset berikutnya perlu setidaknya satu `Survey_Date` yang merepresentasikan tanggal pengukuran/kunjungan.

### Antropometri

Tidak terdapat umur, tinggi, berat, atau z-score. Maka tidak mungkin memeriksa:

- age-heaping atau age misreporting;
- biological plausibility age-height-weight;
- height-for-age z-score (HAZ);
- weight-for-age z-score (WAZ);
- weight-for-height z-score (WHZ);
- flag outlier WHO;
- konsistensi antara status stunting dan HAZ;
- rounding/terminal digit preference pada pengukuran.

Untuk data antropometri, pipeline yang ideal adalah melakukan validasi pada nilai mentah **sebelum** menerima z-score yang disediakan sumber data, kemudian membandingkan z-score sumber versus z-score yang dihitung ulang.

---

## 12. Advanced analyses yang direkomendasikan setelah dataset diperbaiki

### 12.1 Missingness mechanism

Pisahkan MCAR/MAR/MNAR assessment. Missingness indicator dapat dimodelkan terhadap distrik, enumerator, tanggal, dan karakteristik responden untuk mendeteksi systematic missingness.

### 12.2 Hierarchical variance decomposition

Gunakan random effects bertingkat untuk mengukur kontribusi relatif distrik, puskesmas, dan enumerator. Ini penting untuk membedakan variasi konteks wilayah dari variasi proses pengumpulan data.

### 12.3 Bayesian measurement-error model

Jika tinggi/berat memiliki rounding atau error pengukuran, model Bayesian dapat memasukkan latent true measurement dan error process. Ini lebih tepat daripada sekadar membuang outlier ekstrem.

### 12.4 Spatial analysis

Jika tersedia koordinat atau polygon distrik/puskesmas, lakukan spatial autocorrelation (Moran's I), local indicators, serta spatial hierarchical model. Jangan menggunakan kode distrik sebagai proxy spasial tanpa geometri aktual.

### 12.5 Robust predictive modeling

Setelah ID individu dibenahi, lakukan GroupKFold berdasarkan anak/cluster, calibration curve, Brier score, PR-AUC, decision-curve style analysis bila konteks keputusan tersedia, dan permutation importance/SHAP dengan pencegahan leakage.

### 12.6 Causal analysis

Untuk klaim programatik, gunakan DAG lebih dulu. Misalnya, kondisi air/sanitasi, paparan asap, pola pemberian makan, akses layanan, status sosial-ekonomi, dan umur dapat membentuk confounding dan mediation structure yang berbeda. Karena data saat ini cross-sectional dan tidak memiliki temporal ordering, causal effect tidak dapat diestimasi secara defensible dari file ini saja.

### 12.7 Sensitivity to unobserved confounding

Jika nanti ingin menginterpretasikan OR/RR sebagai efek program, lakukan quantitative bias analysis atau sensitivity analysis terhadap unmeasured confounding daripada hanya mengandalkan p-value.

---

## 13. Quality-control rules yang sebaiknya diterapkan pada pipeline berikutnya

1. Setiap anak harus mempunyai `Child_ID` unik dan stabil.
2. `Enumerator_ID`, `Puskesmas_ID`, dan `Survey_Date` wajib hadir.
3. Tandai record exact-duplicate dan near-duplicate, tetapi jangan langsung menghapus tanpa aturan bisnis.
4. Audit domain values untuk semua binary fields dan dokumentasikan arti 0/1.
5. Dokumentasikan `Malnutrition_Level` dengan codebook resmi.
6. Simpan raw measurement dan derived z-score secara terpisah.
7. Recalculate HAZ/WAZ/WHZ dari umur, sex, height, dan weight lalu bandingkan dengan sumber.
8. Implementasikan automated data-quality report pada setiap batch pengumpulan data.
9. Validasi train/test split agar tidak ada child/cluster leakage.
10. Semua analisis utama harus memiliki sensitivity analysis terhadap duplicate handling dan missing-data handling.

---

## 14. Kesimpulan statistik

Data saat ini memberikan bukti deskriptif yang kuat mengenai perbedaan prevalence stunting antar distrik dan sejumlah asosiasi dengan variabel biner. Namun, **temuan terpenting bukan angka OR tertentu melainkan struktur data**: tidak ada identifier individu dan terdapat banyak exact duplicate records.

Akibatnya, estimasi prevalence dan asosiasi cukup sensitif terhadap cara duplikasi diperlakukan. `Malnutrition_Level` juga menunjukkan struktur yang sangat dekat dengan outcome sehingga sangat berisiko menjadi derived label atau target leakage.

Secara metodologis, dataset saat ini paling aman diperlakukan sebagai **cross-sectional categorical survey extract untuk exploratory analysis**, bukan sebagai dataset siap untuk inference kausal, evaluasi kinerja petugas, trend surveillance, atau validasi antropometri.

Versi berikutnya yang menyertakan ID anak, enumerator/puskesmas/perawat, tanggal, umur-bulan, tinggi, berat, dan codebook akan memungkinkan analisis yang jauh lebih kuat: hierarchical Bayesian model penuh, temporal surveillance, quality-control per petugas, measurement-error model, serta validasi antropometri yang benar-benar berbasis WHO/reference standard.

---

## 15. Reproducibility

Analisis dibuat dalam Python terpisah dari report ini. Script utama:

`mamberamo_stunting_analysis.py`

Jalankan:

```bash
python mamberamo_stunting_analysis.py --input DataProc.csv --output mamberamo_analysis_output
```

Output utama mencakup CSV ringkasan, JSON model diagnostics, Bayesian district partial pooling, Bayesian mixed-effects logistic regression, duplicate sensitivity analysis, dan tiga figure.
