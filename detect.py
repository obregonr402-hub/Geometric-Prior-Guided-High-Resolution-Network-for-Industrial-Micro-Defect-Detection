import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# 预测框粗细和颜色修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点

if __name__ == '__main__':
    model = RTDETR(r'C:\Users\86181\Desktop\RTDETR-main\runsx\train\pcb\rtdetr-A2\weights\best.pt') # select your model.pt path
    abbreviations = {
        'rolled-in_scale': 'Roll',
    }

    # 遍历并替换名字
    for key, val in list(model.names.items()):
        if val in abbreviations:
            new_name = abbreviations[val]
            # 1. 改外层包装器的名字
            model.names[key] = new_name
            # 2. 改底层 PyTorch 模型的名字 (真正起作用的这句！)
            if hasattr(model, 'model') and hasattr(model.model, 'names'):
                model.model.names[key] = new_name
    model.predict(source=r'C:\Users\86181\Desktop\RTDETR-main\datasets_pcb完整 - 加工版模糊版\images\val',
                  conf=0.25,
                  project='runs/detect',
                  name='base',
                  save=True,
                  line_width=2,  # 【已开启】线宽变细，防止挡住缺陷
                  # show_conf=False,  # 【已开启】不显示概率值 (0.85)，让图面干净
                  # show_labels=False,  # 如果你连 RS 都不想要，就把这行的 # 也删掉！
                  save_txt=True
                  #save=True,
                  # visualize=True # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )