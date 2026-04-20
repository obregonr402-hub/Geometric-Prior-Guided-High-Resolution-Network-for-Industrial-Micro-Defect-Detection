import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR
from ultralytics import YOLO


if __name__ == '__main__':
    model = RTDETR(r'C:\Users\86181\Desktop\RTDETR-main\ultralytics\cfg\models\rt-detr\ABC.yaml')
    # model.load('') # loading pretrain weights
    model.train(data=r'C:\Users\86181\Desktop\RTDETR-main\datasets_pcb完整 - 加工版模糊版\data.yaml',
                cache=False,
                imgsz=640,
                epochs=1,
                batch=16, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
                workers=2, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0,1', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
                #resume=True, # last.pt path
                patience=0, # 设置0代表不早提供，设置30代表精度持续30epoch没有比之前最高的高就早停
                project='runs/train',
                name='pcb/B',
                )