import json
from collections import defaultdict

desc_path = "./descriptions/full_descriptions.json"

with open(desc_path, "r") as f:
    data = json.load(f)

total_demos = 0
unique_file_paths = set()
demo_counts_per_file = defaultdict(int)
empty_caption_count = 0

for file_path, demos in data.items():
    unique_file_paths.add(file_path)
    for demo_id, content in demos.items():
        total_demos += 1
        demo_counts_per_file[file_path] += 1

        caption = content.get("caption", "").strip()
        if caption == "":
            empty_caption_count += 1

# 출력
print(f"📦 총 trajectory 수 (demo_id): {total_demos} / 기대: 3917") # 3917
print(f"📚 고유 task 종류 수 (file_path): {len(unique_file_paths)} / 기대: 89") # 89
print(f"📄 caption이 비어있는 trajectory 수: {empty_caption_count}")
print(f"\n📊 task별 demo 개수 상위 5개:")
for k in sorted(demo_counts_per_file, key=demo_counts_per_file.get, reverse=True)[:5]:
    print(f"  - {k}: {demo_counts_per_file[k]}")
