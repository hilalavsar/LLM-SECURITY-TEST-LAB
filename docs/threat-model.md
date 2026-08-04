# Tehdit Modeli (Threat Model)

> Iskelet — birlikte doldurulacak.

## Neyi koruyoruz?
- Sistem prompt icindeki gizli talimatlar / canary degeri.
- Modelin guvenli davranis sinirlari.

## Saldirgan kim?
- Uygulamaya prompt gonderebilen son kullanici.

## Saldiri siniflari (OWASP LLM temelli)
- Direct prompt injection
- Jailbreak / role-play
- System-prompt extraction
- Canary / secret leakage
- Obfuscation (kodlama, harf bolme)
- Indirect injection (dolayli)
- Multi-turn

## Basari kriteri (saldirgan acisindan)
- Canary'nin dogrudan veya dolayli sizmasi.
- Modelin yasak eylemi gerceklestirmesi.

## Bizim savunmalarimiz (Hafta 6)
- Defense 1: sertlestirilmis sistem prompt sablonu.
- Defense 2: girdi risk skorlayici.
- (ops.) cikti canary kontrolu.
