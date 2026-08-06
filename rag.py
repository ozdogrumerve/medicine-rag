from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from google import genai
from typer import prompt
import pickle

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def pdf_oku(dosya_yolu: str) -> str:
    """
    Verilen PDF dosyasının tüm sayfalarını okuyup
    tek bir metin (string) olarak döndürür.
    """
    reader = PdfReader(dosya_yolu)
    tum_metin = ""

    for sayfa in reader.pages:
        tum_metin += sayfa.extract_text() + "\n"

    return tum_metin


def metni_boles(metin: str, chunk_boyutu: int = 500, overlap: int = 50) -> list[str]:
    """
    Uzun metni, aralarında biraz örtüşme (overlap) olan
    küçük parçalara böler.
    """
    parcalar = []
    baslangic = 0

    while baslangic < len(metin):
        bitis = baslangic + chunk_boyutu
        parca = metin[baslangic:bitis]
        parcalar.append(parca)
        baslangic += chunk_boyutu - overlap  # overlap kadar geri kayarak devam et

    return parcalar


# Modeli bir kere yüklüyoruz, tüm program boyunca bunu kullanacağız
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def embedding_olustur(parcalar: list[str]):
    """
    Metin parçalarını vektöre çevirir ve
    FAISS index'ine kaydeder.
    """
    vektorler = model.encode(parcalar)
    vektorler = np.array(vektorler).astype("float32")

    boyut = vektorler.shape[1]  # her vektörün kaç sayıdan oluştuğu
    index = faiss.IndexFlatL2(boyut)
    index.add(vektorler)

    return index


def index_kaydet(index, parcalar: list[str], klasor: str = "veritabani"):
    """
    FAISS index'ini ve chunk listesini diske kaydeder.
    """
    os.makedirs(klasor, exist_ok=True)
    faiss.write_index(index, f"{klasor}/index.faiss")

    with open(f"{klasor}/parcalar.pkl", "wb") as f:
        pickle.dump(parcalar, f)


def index_yukle(klasor: str = "veritabani"):
    """
    Diskten FAISS index'ini ve chunk listesini okur.
    Eğer dosyalar yoksa None döndürür.
    """
    index_yolu = f"{klasor}/index.faiss"
    parca_yolu = f"{klasor}/parcalar.pkl"

    if not os.path.exists(index_yolu) or not os.path.exists(parca_yolu):
        return None, None

    index = faiss.read_index(index_yolu)

    with open(parca_yolu, "rb") as f:
        parcalar = pickle.load(f)

    return index, parcalar


def ilgili_parcalari_bul(soru: str, index, parcalar: list[dict], k: int = 5, kaynak_filtre: str = None):
    """
    Kullanıcının sorusunu vektöre çevirir, FAISS'te
    en yakın k tane parçayı bulup döndürür.
    kaynak_filtre verilirse, sadece o dosyaya ait sonuçlar döner.
    """
    soru_vektoru = model.encode([soru])
    soru_vektoru = np.array(soru_vektoru).astype("float32")

    # Filtre varsa daha fazla sonuç çekip sonra filtreleyeceğiz
    arama_k = len(parcalar) if kaynak_filtre else k

    mesafeler, indeksler = index.search(soru_vektoru, arama_k)

    bulunan_parcalar = [parcalar[i] for i in indeksler[0]]

    if kaynak_filtre:
        bulunan_parcalar = [p for p in bulunan_parcalar if p["kaynak"] == kaynak_filtre]
        bulunan_parcalar = bulunan_parcalar[:k]

    return [p["metin"] for p in bulunan_parcalar]


def cevap_uret(soru: str, ilgili_parcalar: list[str]) -> str:
    """
    Bulunan parçaları bağlam olarak Gemini'ye gönderir,
    soruya bu bağlama dayanarak cevap vermesini ister.
    """
    baglam = "\n\n".join(ilgili_parcalar)

    prompt = f"""Sen bir ilaç prospektüsü asistanısın. Aşağıda bir ilaç prospektüsünden alınmış parçalar var.
Sadece bu parçalardaki bilgiye dayanarak soruyu cevapla.
Eğer cevap bu parçaların içinde yoksa, "Bu bilgi prospektüste bulunamadı, lütfen doktorunuza veya eczacınıza danışın." de.
Tıbbi tavsiye verme, sadece prospektüste yazanı aktar.

BAĞLAM:
{baglam}

SORU: {soru}

CEVAP:"""

    yanit = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )

    return yanit.text

# Hızlı test için main bloğu
if __name__ == "__main__":
    index, parcalar = index_yukle()

    soru = "NUROFEN nedir?"
    sonuclar = ilgili_parcalari_bul(soru, index, parcalar, k=200)

    for i, parca in enumerate(sonuclar):
        if "ibuprofen" in parca.lower() and "nurofen" in parca.lower():
            print(f"Etkin madde chunk'ı {i+1}. sırada bulundu")
            print(parca[:300])
            break