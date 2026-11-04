# Edge Impulse - OpenMV FOMO Object Detection Example
#
# This work is licensed under the MIT license.
# Copyright (c) 2013-2024 OpenMV LLC. All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE

import sensor, image, time, os, ml, math, uos, gc
from ulab import numpy as np
from pyb import UART, Pin, Timer


sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((240,240))
sensor.skip_frames(time = 2000)
sensor.set_auto_gain(True) # must be turned off for color tracking
sensor.set_auto_whitebal(True) # must be turned off for color tracking

uart = UART(3, 115200)  # 设置串口波特率，与stm32一致
uart.init(115200, bits=8, parity=None, stop=1)

tim = Timer(4, freq=1000)  # Frequency in Hz
led_dac = tim.channel(1, Timer.PWM, pin=Pin("P7"), pulse_width_percent=50)
led_dac.pulse_width_percent(0)
led_status = 0

move_x = 0
move_y = 0
movex_threshold = 30
movey_threshold = 20
target_cx = 120
target_cy = 180
status_time = 0
fans_flag = False
fans_time = 0
status = 0
servo0 = 1500
servo1 = 1150


net = None
labels = None
min_confidence = 0.5

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
    (255,   0,   0),
    (  0, 255,   0),
    (255, 255,   0),
    (  0,   0, 255),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 255, 255),
]

threshold_list = [(math.ceil(min_confidence * 255), 255)]

def fomo_post_process(model, inputs, outputs):
    ob, oh, ow, oc = model.output_shape[0]

    x_scale = inputs[0].roi[2] / ow
    y_scale = inputs[0].roi[3] / oh

    scale = min(x_scale, y_scale)

    x_offset = ((inputs[0].roi[2] - (ow * scale)) / 2) + inputs[0].roi[0]
    y_offset = ((inputs[0].roi[3] - (ow * scale)) / 2) + inputs[0].roi[1]

    l = [[] for i in range(oc)]

    for i in range(oc):
        img = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img.find_blobs(
            threshold_list, x_stride=1, y_stride=1, area_threshold=1, pixels_threshold=1
        )
        for b in blobs:
            rect = b.rect()
            x, y, w, h = rect
            score = (
                img.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            )
            x = x * scale + x_offset
            y = y * scale + y_offset
            w = w * scale
            h = h * scale
            l[i].append((x, y, w, h, score))
    return l

#计算左右转速度
def speed_omega(nearest_cx):
    speed = max(0.05, abs(nearest_cx - target_cx) / 600)
    return speed

#计算前后转速度
def speed_front(nearest_cy):
    speed = max(0.05,abs(nearest_cy - target_cy) / 300)
    return speed

clock = time.clock()

run_status = 0
#移动摄像头舵机
uart.write("{{#000P{:0>4d}T1100!#001P{:0>4d}T1100!}}\n".format(servo0, servo1))
time.sleep_ms(1000)
while(True):
    clock.tick()
    img = sensor.snapshot()
    img.lens_corr(strength=1.8, zoom=1.0)
    # print(clock.fps(), "fps", end="\n\n")

    if uart.any():  # 接收字符串
        try:  # 用来判断串口数据异常
            string = uart.read()
            if string:
                string = string.decode()
                # print(string)
                if string.find("$TurnonLed!") >= 0:  # 开灯指令
                    led_dac.pulse_width_percent(100)
                elif string.find("$TurnoffLed!") >= 0:  # 关灯指令
                    led_dac.pulse_width_percent(0)
                elif string.find("$StopRun!") >= 0:  # 停止所有运行并复位
                    led_dac.pulse_width_percent(0)
                    run_status = 0
                    fans_flag = False
                    # print(f"{run_status}")
                    uart.write("$Car:0,0,0,0!\n")   # 停车
                    time.sleep_ms(50)
                elif string.find("$StartRun!") >= 0:  # 开始运行
                    run_status = 1
                    # print(f"{run_status}\n")
        except:
            continue

    if run_status == 0:
        continue

    nearest_target = (120,180,0)
    min_distance = float('inf')
    for i, detection_list in enumerate(net.predict([img], callback=fomo_post_process)):
        if i == 0: continue  # background class
        if len(detection_list) == 0:
            status = 0
            status_time = time.time()
            continue  # no detections for this class?
        #print("********** %s **********" % labels[i])
        status = 1
        for x, y, w, h, score in detection_list:
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            #print(f"x {center_x}\ty {center_y}\tscore {score}")
            img.draw_circle((center_x, center_y, 15), color=colors[6])
            img.draw_string(math.floor(x), math.floor(y - 15), f"{score:.2f}", color=colors[6],mono_space=False)
            distance_sq = (center_x - target_cx)**2 + (center_y - target_cy)**2
            if min_distance > distance_sq:
                min_distance = distance_sq
                nearest_target = (center_x,center_y,score)
        print('x  ' , nearest_target[0], 'y  ',nearest_target[1])
            # 找到最靠近中心点的坐标
        # ************************运动车**********************************
    # 没有检索到目标
    if status == 0:
        # 风机打开超时 关闭风机
        if(fans_flag):
            if (time.time() - fans_time > 20):
                uart.write("$GBFJ!\n")
                time.sleep_ms(50)
                print('close fans!\n')
                fans_flag = False
        uart.write("$Car:0.05,-0.05,0.05,-0.05!\n")
        print('Stop!\n')
    # 检索到目标
    if status == 1:
        # 更新风机起始时间
        fans_time = time.time()
        if(fans_flag == False):
            uart.write("$DKFJ!\n")
            fans_flag = True
            print('Open fans!\n')
            time.sleep_ms(50)
        # 左右移动
        if abs(nearest_target[0] - target_cx) >= movex_threshold:
            status_time = time.time()
            speedx = speed_omega(nearest_target[0])       # 计算速度

            if nearest_target[0] > target_cx:
                uart.write("$Car:{},{},{},{}!\n".format(speedx,-speedx,speedx,-speedx))
                print('turn right!')
            else:
                uart.write("$Car:{},{},{},{}!\n".format(-speedx,speedx,-speedx,speedx))
                print('turn left!')
        # 前后移动
        elif abs(nearest_target[1] - target_cy) >= movey_threshold:
            status_time = time.time()
            speedy = speed_front(nearest_target[1])
            if nearest_target[1] > target_cy:
                uart.write("$Car:{},{},{},{}!\n".format(-speedy,-speedy,-speedy,-speedy))
                print('back!')
            else:
                uart.write("$Car:{},{},{},{}!\n".format(speedy,speedy,speedy,speedy))
                print('front!')
        else:
            status = 2
    if status == 2:
        uart.write("$Car:0,0,0,0!\n")
        print('Stop!\n')
        time.sleep_ms(50)
        current_time = time.time()
        # 保持中心位置超过2s
        if current_time - status_time >= 2:
            status_time = current_time
            uart.write("$Car:0.18,0.18,0.18,0.18!\n")
            time.sleep_ms(1000)


