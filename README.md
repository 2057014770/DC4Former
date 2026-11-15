# DC4Former

## File structure

```text
dc4former.py                         DC4Former backbone
test_rotation.py                     Eight-angle and multi-angle evaluation script
configs/                             FloodNet and RescueNet configurations
FloodNet_DC4Former.pth               FloodNet checkpoint
RescueNet_DC4Former.pth              RescueNet checkpoint
```

## Dataset structure

The configurations expect the following dataset roots:

```text
data/
  FloodNet/
    img_dir/{train,val,test}/
    ann_dir/{train,val,test}/
  RescueNet/
    img_dir/{train,val,test}/
    ann_dir/{train,val,test}/
```
