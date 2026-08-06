from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os

from rag import (
    pdf_oku,
    metni_boles,
    embedding_olustur,
    ilgili_parcalari_bul,
    cevap_uret,
    index_kaydet,
    index_yukle,
)

app = FastAPI()

# Uygulama başlarken, diskte kayıtlı bir index varsa onu yükle.
index, parcalar = index_yukle()
if parcalar is None:
    parcalar = []

# Yüklenmiş ilaçların isim listesi (kaynak filtreleme için)
yuklu_ilaclar = list({p["kaynak"] for p in parcalar}) if parcalar else []


class SoruIstek(BaseModel):
    soru: str


@app.post("/upload")
def upload(dosya: UploadFile = File(...)):
    global index, parcalar, yuklu_ilaclar

    # Dosya adından ilaç ismini çıkar (örn. "NUROFEN.pdf" -> "NUROFEN")
    ilac_adi = os.path.splitext(dosya.filename)[0].upper()

    # Yüklenen dosyayı geçici olarak diske kaydet
    gecici_yol = f"data/prospectus/{dosya.filename}"
    with open(gecici_yol, "wb") as f:
        shutil.copyfileobj(dosya.file, f)

    # PDF'i oku, chunk'la, embed'le
    metin = pdf_oku(gecici_yol)
    ham_parcalar = metni_boles(metin)

    # Her chunk'ı kaynak bilgisiyle birlikte sözlük hâline getir
    yeni_parcalar = [{"metin": p, "kaynak": ilac_adi} for p in ham_parcalar]

    # Embedding sadece düz metinlerle çalıştığı için, metinleri ayrıca çıkarıyoruz
    yeni_index = embedding_olustur(ham_parcalar)

    if index is None:
        index = yeni_index
        parcalar = yeni_parcalar
    else:
        index.merge_from(yeni_index)
        parcalar = parcalar + yeni_parcalar

    if ilac_adi not in yuklu_ilaclar:
        yuklu_ilaclar.append(ilac_adi)

    index_kaydet(index, parcalar)

    return {
        "mesaj": f"{dosya.filename} başarıyla yüklendi ve işlendi",
        "eklenen_parca_sayisi": len(yeni_parcalar),
        "toplam_parca_sayisi": len(parcalar),
    }


@app.post("/ask")
def ask(istek: SoruIstek):
    if index is None:
        return {"hata": "Henüz hiç prospektüs yüklenmedi. Önce /upload ile bir PDF yükleyin."}

    soru_buyuk = istek.soru.upper()
    kaynak_filtre = None
    for ilac in yuklu_ilaclar:
        if ilac in soru_buyuk:
            kaynak_filtre = ilac
            break

    if kaynak_filtre:
        # O ilaca ait toplam chunk sayısını bul, hepsini kullan (üst sınır 40)
        ilgili_chunk_sayisi = sum(1 for p in parcalar if p["kaynak"] == kaynak_filtre)
        kullanilacak_k = min(ilgili_chunk_sayisi, 40)
    else:
        kullanilacak_k = 8

    ilgili_parcalar = ilgili_parcalari_bul(istek.soru, index, parcalar, k=kullanilacak_k, kaynak_filtre=kaynak_filtre)
    cevap = cevap_uret(istek.soru, ilgili_parcalar)
    return {"soru": istek.soru, "cevap": cevap, "kullanilan_filtre": kaynak_filtre, "kullanilan_k": kullanilacak_k}