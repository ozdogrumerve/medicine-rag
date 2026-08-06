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
# Yoksa boş başla, ilk /upload ile oluşturulacak.
index, parcalar = index_yukle()


class SoruIstek(BaseModel):
    soru: str


@app.post("/upload")
def upload(dosya: UploadFile = File(...)):
    global index, parcalar

    # Yüklenen dosyayı geçici olarak diske kaydet
    gecici_yol = f"data/prospectus/{dosya.filename}"
    with open(gecici_yol, "wb") as f:
        shutil.copyfileobj(dosya.file, f)

    # PDF'i oku, chunk'la, embed'le
    metin = pdf_oku(gecici_yol)
    yeni_parcalar = metni_boles(metin)
    yeni_index = embedding_olustur(yeni_parcalar)

    if index is None:
        # İlk yükleme, sıfırdan başlıyoruz
        index = yeni_index
        parcalar = yeni_parcalar
    else:
        # Mevcut index'e yeni vektörleri ekle
        index.merge_from(yeni_index)
        parcalar = parcalar + yeni_parcalar

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

    ilgili_parcalar = ilgili_parcalari_bul(istek.soru, index, parcalar, k=5)
    cevap = cevap_uret(istek.soru, ilgili_parcalar)
    return {"soru": istek.soru, "cevap": cevap}