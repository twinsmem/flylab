import json, random
random.seed(1)
px = [0.0] * 256
for i in random.sample(range(256), 40):
    px[i] = 0.9
print(json.dumps({"pixels": px}))
