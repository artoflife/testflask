# 🌊 Flood Evacuation WebGIS — Cilacap
**Flask + ORS + OSRM + BMKG + Random Forest**

## File yang Dibutuhkan
Sebelum deploy, pastikan kamu punya file-file ini:
```
model/random_forest_model.pkl
data/Desa_Rawan_Banjir.csv
data/Tempat_Evakuasi_Final__1_.csv
static/index.html  ← sudah ada
app.py             ← sudah ada
```

---

## 🚀 Deploy Gratis ke Railway (Paling Mudah)

1. **Buat akun** di https://railway.app (login pakai GitHub)

2. **Push ke GitHub dulu:**
   ```bash
   git init
   git add .
   git commit -m "first commit"
   git branch -M main
   git remote add origin https://github.com/USERNAME/REPO.git
   git push -u origin main
   ```

3. **Di Railway:**
   - Klik **New Project → Deploy from GitHub repo**
   - Pilih repo kamu
   - Railway otomatis detect `Procfile` dan langsung deploy

4. **Set environment variable** (opsional, kalau punya ORS key):
   - Di Railway dashboard → Variables → tambah:
     ```
     ORS_API_KEY=your_key_here
     ```

5. Selesai! Railway kasih URL publik gratis (format: `xxx.up.railway.app`)

---

## 🚀 Deploy Gratis ke Render

1. **Buat akun** di https://render.com

2. Push ke GitHub (sama seperti langkah di atas)

3. **Di Render:**
   - Klik **New → Web Service**
   - Connect GitHub repo
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
   - Pilih plan **Free**

4. Set environment variable `ORS_API_KEY` jika ada

---

## ⚙️ Jalankan Lokal

```bash
pip install -r requirements.txt
python app.py
# Buka http://localhost:5000
```

---

## 🔑 ORS API Key (Opsional tapi Direkomendasikan)

Tanpa key → routing pakai OSRM (real road, gratis unlimited)  
Dengan key → routing pakai ORS (real road + **hindari zona banjir**)

Daftar gratis di: https://openrouteservice.org/dev/#/signup  
(2000 request/hari, tidak perlu kartu kredit)

Set sebagai environment variable:
```
ORS_API_KEY=eyJhbGc...
```

---

## Routing Chain
```
ORS (avoid flood zones) → OSRM (fallback) → Haversine (last resort)
```
