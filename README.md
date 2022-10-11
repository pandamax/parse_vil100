# datasets structure:
```
VIL-100
    |----Annotations
    |----data
    |----JPEGImages
    |----Json
```


# parse_vil100
for parsing and converting dataset vil100.

- vis_vil.py: visualize datasets on original image,incude points and curves form.
- vil2mask.py：generate lane instance mask.
- vil2tusimples.py: convert datasets to tusimple-like format.
- vis_converted.py: visualize converted tusimple-like format.

the scripts above have tested on wsl/linux.

# reference
[VIL-100 Dataset: A Large Annotated Dataset of Video Instance Lane Detection](https://github.com/yujun0-0/MMA-Net/tree/main/dataset)
