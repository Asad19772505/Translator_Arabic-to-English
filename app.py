# ------------ Safer translation core with multiple fallbacks ------------
def translate_text(
    text: str,
    src_lang: str = "auto",
    tgt_lang: str = "en",
    backend: str = "auto",
    libre_url: str | None = None,
    libre_api_key: str | None = None,
):
    """
    Try the selected backend, then fall back through others.
    Returns: (translated_text, engine_used)
    Raises: RuntimeError if all backends fail.
    """
    errors = []

    def try_googletrans():
        from googletrans import Translator
        # Use endpoints that are often less rate-limited
        translator = Translator(service_urls=["translate.googleapis.com", "translate.google.com"])
        res = translator.translate(text, src=src_lang, dest=tgt_lang)
        return res.text, "googletrans"

    def try_deep_google():
        from deep_translator import GoogleTranslator
        # timeout helps avoid hanging
        res = GoogleTranslator(source=src_lang, target=tgt_lang, timeout=7).translate(text)
        return res, "deep-google"

    def try_deep_libre():
        # Works with public instance or a self-hosted server
        from deep_translator import LibreTranslator
        base = libre_url or "https://libretranslate.com"
        res = LibreTranslator(
            source=src_lang if src_lang != "auto" else "auto",
            target=tgt_lang,
            api_key=libre_api_key,
            base_url=base,
            timeout=10,
        ).translate(text)
        return res, f"libretranslate@{base}"

    def try_deep_mymemory():
        from deep_translator import MyMemoryTranslator
        res = MyMemoryTranslator(source=src_lang if src_lang != "auto" else "auto",
                                 target=tgt_lang, timeout=7).translate(text)
        return res, "mymemory"

    order_map = {
        "googletrans": [try_googletrans],
        "deep-google": [try_deep_google],
        "libre":       [try_deep_libre],
        "mymemory":    [try_deep_mymemory],
        "auto":        [try_googletrans, try_deep_google, try_deep_libre, try_deep_mymemory],
    }
    for fn in order_map.get(backend, order_map["auto"]):
        try:
            return fn()
        except Exception as e:
            errors.append(f"{fn.__name__}: {e}")

    raise RuntimeError("All translators failed:\n" + "\n".join(errors))
