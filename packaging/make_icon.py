"""Draw the program icon (PNG 1024 px with alpha, Windows .ico, macOS .icns).

A cell seen in the microscope on the dark background of the program: green actin cortex, blue
lobed nucleus, magenta target spots, in a rounded square in the accent blue. Readable at 16 px
because the ring and the nucleus are large and high in contrast.

    python packaging/make_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
S = 1024
K = 4                                      # supersampling


def draw() -> Image.Image:
    n = S * K
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(0.04 * n)
    d.rounded_rectangle([pad, pad, n - pad, n - pad], radius=int(0.2 * n), fill=(30, 34, 39, 255),
                        outline=(42, 100, 150, 255), width=int(0.035 * n))
    c = n // 2
    r = int(0.33 * n)
    glow = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([c - r, c - r, c + r, c + r], outline=(43, 212, 107, 255), width=int(0.075 * n))
    glow = glow.filter(ImageFilter.GaussianBlur(n * 0.012))
    img.alpha_composite(glow)
    d = ImageDraw.Draw(img)
    # lobed nucleus: three overlapping discs
    for dx, dy, rr in ((-0.07, -0.05, 0.14), (0.08, -0.02, 0.13), (-0.01, 0.1, 0.12)):
        x, y, q = c + int(dx * n), c + int(dy * n), int(rr * n)
        d.ellipse([x - q, y - q, x + q, y + q], fill=(61, 125, 255, 255))
    # target spots
    for dx, dy in ((0.2, 0.16), (0.14, 0.24), (-0.2, 0.18), (0.23, -0.15), (-0.17, -0.22), (0.02, -0.25)):
        x, y, q = c + int(dx * n), c + int(dy * n), int(0.033 * n)
        d.ellipse([x - q, y - q, x + q, y + q], fill=(255, 43, 214, 255))
    return img.resize((S, S), Image.LANCZOS)


def main() -> None:
    img = draw()
    png = ROOT / "localization_tool" / "resources" / "icon.png"
    img.save(png)
    img.save(ROOT / "packaging" / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                                                      (128, 128), (256, 256)])
    try:
        img.save(ROOT / "packaging" / "icon.icns")
    except Exception as exc:  # Pillow writes ICNS on all platforms in recent versions
        print("icns not written:", exc)
    print("written", png)


if __name__ == "__main__":
    main()
