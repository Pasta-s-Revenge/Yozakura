from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, copy_metadata

ROOT = Path(SPECPATH).parent.parent

packages = (
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
)

datas = []
binaries = []
hiddenimports = []

for package in packages:
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports
    datas += copy_metadata(package)

# Transformers loads these resources dynamically.
datas += collect_data_files("transformers", include_py_files=False)
binaries += collect_dynamic_libs("torch")

analysis = Analysis(
    [str(ROOT / "src" / "yozakura" / "cli.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="yozakura",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="yozakura",
)
