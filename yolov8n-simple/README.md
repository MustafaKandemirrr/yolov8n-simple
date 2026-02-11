# yolov8n-simple

Node.js ile ONNX formatındaki YOLOv8n modelini kullanarak tek bir görsel üzerinde arac tespiti yapan, sonucunda da otopark doluluk oranini hesaplayan basit bir proje.

Bu proje:
- `onnxruntime-node` ile modeli calistirir,
- `sharp` ile gorseli modele uygun bicime getirir,
- ham model ciktilarindan kutulari cikarir,
- Non-Maximum Suppression (NMS) uygular,
- arac sayisi ve doluluk oranini terminalde raporlar,
- asama bazli sure ve bellek kullanim metriklerini yazdirir.

## Icerik
- [Ozellikler](#ozellikler)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Hizli Baslangic](#hizli-baslangic)
- [Calisma Mantigi](#calisma-mantigi)
- [Yapilandirma](#yapilandirma)
- [Ornek Cikti](#ornek-cikti)
- [Proje Yapisi](#proje-yapisi)
- [Sorun Giderme](#sorun-giderme)
- [Gelisim Fikirleri](#gelisim-fikirleri)
- [Lisans](#lisans)

<a id="ozellikler"></a>
## Ozellikler
- YOLOv8n ONNX modeliyle arac tespiti
- Tek goruntu uzerinden analiz
- Confidence threshold ve IoU threshold ayari
- NMS ile tekrar eden kutularin elenmesi
- Toplam kapasiteye gore doluluk orani hesabi
- Performans raporu:
  - model yukleme suresi
  - on isleme suresi
  - model cikarim suresi
  - kutu cikarim suresi
  - NMS suresi
  - toplam sure
  - bellek kullanim ozeti (RSS, Heap, External)

<a id="gereksinimler"></a>
## Gereksinimler
- Node.js (LTS onerilir)
- npm
- `yolov8n.onnx` model dosyasi
- Analiz edilecek gorsel dosyasi (`araba.jpg` varsayilan)

> Not: Kod varsayilan olarak proje kokunde `yolov8n.onnx` ve `araba.jpg` dosyalarini arar.

<a id="kurulum"></a>
## Kurulum
1. Depoyu klonlayin:

```bash
git clone <REPO_URL>
cd yolov8n-simple
```

2. Bagimliliklari kurun:

```bash
npm install
```

3. Model dosyasini proje kokune koyun:
- `yolov8n.onnx`

4. Girdi gorselini proje kokune koyun:
- `araba.jpg` (veya koddan baska bir dosya adi tanimlayin)

<a id="hizli-baslangic"></a>
## Hizli Baslangic
Projeyi calistirmak icin:

```bash
node index.js
```

Calisma sonunda terminalde su bilgiler gorulur:
- Tespit edilen arac sayisi
- Doluluk orani (yuzde)
- Durum bilgisi (`MUSAIT`, `YOGUN`, `TAMAMEN DOLU`)
- Performans ve bellek raporu

<a id="calisma-mantigi"></a>
## Calisma Mantigi
`index.js` dosyasindaki is akisinin ozeti:

1. **Model yukleme**
   - `onnxruntime-node` ile `yolov8n.onnx` yuklenir.

2. **On isleme**
   - Gorsel `320x320` boyutuna getirilir.
   - Alfa kanali temizlenir.
   - RGB verisi normalize edilerek `Float32Array` icine alinır.
   - Tensor boyutu: `[1, 3, 320, 320]`

3. **Inference (cikarim)**
   - Girdi tensoru modelin `images` girisine verilir.
   - Cikis olarak `output0` okunur.

4. **Kutu cikarimi**
   - Anchor bazli skorlar okunur.
   - `CONF_THRESHOLD` ustundeki aday kutular saklanir.

5. **NMS**
   - Yuksek IoU degerine sahip cakisan kutular elenir.

6. **Raporlama**
   - Son kutu sayisi arac sayisi olarak kabul edilir.
   - `TOTAL_CAPACITY` ile doluluk hesaplanir.
   - Sure ve bellek metrikleri yazdirilir.

<a id="yapilandirma"></a>
## Yapilandirma
`index.js` icinde asagidaki sabitleri degistirerek davranisi ozellestirebilirsiniz:

- `MODEL_PATH`: ONNX model dosya yolu
- `INPUT_IMAGE`: analiz edilecek gorsel yolu
- `CONF_THRESHOLD`: minimum guven skoru (varsayilan: `0.45`)
- `IOU_THRESHOLD`: NMS IoU esigi (varsayilan: `0.50`)
- `TOTAL_CAPACITY`: toplam park kapasitesi (varsayilan: `20`)

### Farkli gorsel ile test etme
`INPUT_IMAGE` satirini degistirebilirsiniz:

```js
const INPUT_IMAGE = path.join(__dirname, 'otopark.jpg');
```

<a id="ornek-cikti"></a>
## Ornek Cikti
Asagidaki gibi bir terminal ciktisi alirsiniz (degerler ortama gore degisir):

```text
========================================
Analiz Tamamlandi
Tespit Edilen Arac: 12
Doluluk Orani: %60.0
DURUM: MUSAIT
========================================

============= PERFORMANS RAPORU =============
Model yukleme       : 35.22 ms
On isleme           : 8.41 ms
Model cikarimi      : 14.09 ms
Kutu cikarimi       : 1.62 ms
NMS                 : 0.53 ms
Toplam sure         : 60.47 ms
=============================================
```

<a id="proje-yapisi"></a>
## Proje Yapisi
```text
yolov8n-simple/
|- index.js
|- package.json
|- package-lock.json
|- yolov8n.onnx
|- araba.jpg
|- README.md
|- node_modules/
```

<a id="sorun-giderme"></a>
## Sorun Giderme
- **`Hata: ... yolov8n.onnx`**
  - Model dosyasi bulunamadi. Dosyanin adini ve konumunu kontrol edin.
- **`Hata: ... araba.jpg`**
  - Girdi gorseli bulunamadi. `INPUT_IMAGE` yolunu duzeltin.
- **Cok dusuk veya cok yuksek arac sayisi**
  - `CONF_THRESHOLD` ve `IOU_THRESHOLD` degerlerini gorselinize gore ayarlayin.
- **Kurulumda native modullerde hata**
  - Node.js surumunu LTS kullanin, `node_modules` silip tekrar `npm install` deneyin.

<a id="gelisim-fikirleri"></a>
## Gelisim Fikirleri
- Cikti gorseline kutulari cizerek gorsel kaydetme
- Komut satiri argumanlariyla (`--image`, `--capacity`) dinamik calisma
- Canli kamera/RTSP akisi destegi
- Sinif filtreleme (sadece car sinifi)
- Birden fazla goruntu icin batch isleme

<a id="lisans"></a>
## Lisans
Bu proje su an icin `ISC` lisansi ile tanimlanmistir (`package.json`).

GitHub'a yuklerken lisans metnini netlestirmek icin bir `LICENSE` dosyasi eklemeniz onerilir.
