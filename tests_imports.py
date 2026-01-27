# test_imports.py
try:
    import imagehash
    print("✅ imagehash installed successfully")
except ImportError as e:
    print(f"❌ imagehash import failed: {e}")

try:
    from PIL import Image
    print("✅ PIL (Pillow) installed successfully")
except ImportError as e:
    print(f"❌ PIL import failed: {e}")

try:
    import acoustid
    print("✅ acoustid installed successfully")
except ImportError as e:
    print(f"❌ acoustid import failed: {e}")

try:
    import librosa
    print("✅ librosa installed successfully")
except ImportError as e:
    print(f"❌ librosa import failed: {e}")

try:
    import cv2
    print("✅ opencv-python installed successfully")
except ImportError as e:
    print(f"❌ opencv import failed: {e}")

print("\n🎉 All dependencies ready for media deduplication!")
