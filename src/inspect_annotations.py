import json, statistics
COCO = r'c:\Users\alexy\Documents\Claude_projects\Kaggle competition\ESA_comp\neural-debris-removal-in-streak-detection-models\unlearn_set\annotations_coco.json'
with open(COCO) as f:
    coco = json.load(f)
print('categories:', coco.get('categories'))
print('num images:', len(coco['images']))
print('num annotations:', len(coco['annotations']))
print()
print('first 5 annotations:')
for ann in coco['annotations'][:5]:
    img = next(i for i in coco['images'] if i['id'] == ann['image_id'])
    x, y, w, h = ann['bbox']
    fname = img['file_name']
    print(f'  img={fname} bbox=[{x:.1f},{y:.1f},{w:.1f},{h:.1f}] area={w*h:.0f} aspect={w/h:.2f}')
print()
bboxes = [ann['bbox'] for ann in coco['annotations']]
widths  = [b[2] for b in bboxes]
heights = [b[3] for b in bboxes]
areas   = [b[2]*b[3] for b in bboxes]
aspects = [b[2]/b[3] for b in bboxes]
xs      = [b[0] for b in bboxes]
ys      = [b[1] for b in bboxes]
print(f'width  : min={min(widths):.1f}  max={max(widths):.1f}  mean={statistics.mean(widths):.1f}  median={statistics.median(widths):.1f}')
print(f'height : min={min(heights):.1f}  max={max(heights):.1f}  mean={statistics.mean(heights):.1f}  median={statistics.median(heights):.1f}')
print(f'area   : min={min(areas):.0f}  max={max(areas):.0f}  mean={statistics.mean(areas):.0f}')
print(f'aspect : min={min(aspects):.2f}  max={max(aspects):.2f}  mean={statistics.mean(aspects):.2f}  median={statistics.median(aspects):.2f}')
print(f'x (TL): min={min(xs):.1f}  max={max(xs):.1f}  mean={statistics.mean(xs):.1f}')
print(f'y (TL): min={min(ys):.1f}  max={max(ys):.1f}  mean={statistics.mean(ys):.1f}')
