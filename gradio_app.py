"""
İlaç Prospektüsü RAG Sistemi - Gradio Arayüzü

Bu script mevcut FastAPI sunucusuna (main.py) HTTP istekleri atar.
Önce FastAPI sunucusunu ayrı bir terminalde çalıştırman gerekiyor:

    uvicorn main:app --reload

Sonra bu dosyayı çalıştır:

    python gradio_app.py

Varsayılan olarak http://127.0.0.1:7860 adresinde açılır.
"""

import gradio as gr
import requests

API_BASE_URL = "http://127.0.0.1:8000"


def api_saglik_kontrolu():
    """FastAPI sunucusunun ayakta olup olmadığını kontrol eder."""
    try:
        requests.get(API_BASE_URL, timeout=2)
        return True
    except requests.exceptions.ConnectionError:
        return False


def pdf_yukle(dosya):
    """Yüklenen PDF'i FastAPI'nin /upload endpoint'ine gönderir."""
    if dosya is None:
        return "⚠️ Lütfen bir PDF dosyası seç."

    if not api_saglik_kontrolu():
        return (
            f"❌ FastAPI sunucusuna bağlanılamadı ({API_BASE_URL}).\n"
            f"Ayrı bir terminalde `uvicorn main:app --reload` çalıştırdığından emin ol."
        )

    try:
        with open(dosya.name, "rb") as f:
            dosya_adi = dosya.name.split("/")[-1].split("\\")[-1]
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files={"dosya": (dosya_adi, f, "application/pdf")},
                timeout=120,
            )
        if response.status_code == 200:
            data = response.json()
            return f"✅ Yükleme başarılı:\n{data}"
        else:
            return f"❌ Hata ({response.status_code}):\n{response.text}"
    except requests.exceptions.RequestException as e:
        return f"❌ İstek hatası: {e}"


def soru_sor(soru):
    """Soruyu FastAPI'nin /ask endpoint'ine gönderir."""
    if not soru or not soru.strip():
        return "⚠️ Lütfen bir soru yaz.", "", ""

    if not api_saglik_kontrolu():
        error_msg = (
            f"❌ FastAPI sunucusuna bağlanılamadı ({API_BASE_URL}).\n"
            f"Ayrı bir terminalde `uvicorn main:app --reload` çalıştırdığından emin ol."
        )
        return error_msg, "", ""

    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"soru": soru},
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            cevap = data.get("cevap", "")
            filtre = data.get("kullanilan_filtre") or "— (filtre uygulanmadı)"
            k = str(data.get("kullanilan_k", ""))
            return cevap, filtre, k
        else:
            return f"❌ Hata ({response.status_code}):\n{response.text}", "", ""
    except requests.exceptions.RequestException as e:
        return f"❌ İstek hatası: {e}", "", ""


with gr.Blocks(title="İlaç Prospektüsü RAG Sistemi") as demo:
    gr.Markdown("# 💊 İlaç Prospektüsü RAG Sistemi")
    gr.Markdown(
        "PDF prospektüs yükle, ardından ilaçlar hakkında Türkçe sorular sor. "
        "Bu arayüz `main.py` içindeki FastAPI sunucusuna bağlanır — sunucunun "
        "çalışıyor olması gerekir (`uvicorn main:app --reload`)."
    )

    with gr.Tab("📄 PDF Yükle"):
        dosya_input = gr.File(label="Prospektüs PDF'i", file_types=[".pdf"])
        yukle_btn = gr.Button("Yükle", variant="primary")
        yukle_output = gr.Textbox(label="Sonuç", lines=4, interactive=False)
        yukle_btn.click(fn=pdf_yukle, inputs=dosya_input, outputs=yukle_output)

    with gr.Tab("❓ Soru Sor"):
        soru_input = gr.Textbox(
            label="Sorunuz",
            placeholder="Örn: NUROFEN günde kaç kez kullanılır?",
            lines=2,
        )
        sor_btn = gr.Button("Sor", variant="primary")
        cevap_output = gr.Textbox(label="Cevap", lines=8, interactive=False)
        with gr.Row():
            filtre_output = gr.Textbox(label="Kullanılan filtre", interactive=False)
            k_output = gr.Textbox(label="Kullanılan k", interactive=False)
        sor_btn.click(
            fn=soru_sor,
            inputs=soru_input,
            outputs=[cevap_output, filtre_output, k_output],
        )
        soru_input.submit(
            fn=soru_sor,
            inputs=soru_input,
            outputs=[cevap_output, filtre_output, k_output],
        )

if __name__ == "__main__":
    demo.launch()
