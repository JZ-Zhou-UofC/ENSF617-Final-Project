from pathlib import Path

p = Path("raw-data/road-sign-detection-DatasetNinja/ds/ann")
print(p.exists())
print(list(p.glob("*"))[:5])