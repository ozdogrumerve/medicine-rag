from fastapi import FastAPI
from pydantic import BaseModel
from rag import pdf_oku, metni_boles, embedding_olustur, ilgili_parcalari_bul, cevap_uret

app = FastAPI()

# Uygulama başlarken PDF'i bir kere okuyup hazırlıyoruz,
# her soruda yeniden okumamak için (performans için)
metin = pdf_oku("data/prospectus/PAROL.pdf")
parcalar = metni_boles(metin)
index = embedding_olustur(parcalar)


class SoruIstek(BaseModel):
    soru: str


@app.post("/ask")
def ask(istek: SoruIstek):
    ilgili_parcalar = ilgili_parcalari_bul(istek.soru, index, parcalar, k=5)
    cevap = cevap_uret(istek.soru, ilgili_parcalar)
    return {"soru": istek.soru, "cevap": cevap}