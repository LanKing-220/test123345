import gc
import sys
import time
import uos

import display
import image
import ml
import sensor


MODEL_PATH = "trained.tflite"
LABELS_PATH = "labels.txt"
LCD_HINT = image.ROTATE_270


def init_camera():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing((240, 240))
    sensor.set_auto_whitebal(False)
    sensor.skip_frames(time=2000)


def init_lcd():
    try:
        return display.SPIDisplay(width=240, height=320)
    except Exception as err:
        print("LCD init failed:", err)
        return None


def show(lcd, img, line1, line2=None, color=(255, 255, 0)):
    if line1:
        img.draw_string(2, 2, line1, color=color, mono_space=False)
    if line2:
        img.draw_string(2, 22, line2, color=color, mono_space=False)
    if lcd is not None:
        lcd.write(img, hint=LCD_HINT)


def smoke_callback(model, inputs, outputs):
    return {
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "output_count": len(outputs),
    }


def main():
    init_camera()
    lcd = init_lcd()

    img = sensor.snapshot()
    show(lcd, img, "Boot OK", "Checking files...")

    try:
        model_size = uos.stat(MODEL_PATH)[6]
        labels = [line.rstrip("\n") for line in open(LABELS_PATH)]
    except Exception as err:
        print("File check failed:", err)
        img = sensor.snapshot()
        show(lcd, img, "FILE FAIL", str(err), color=(255, 0, 0))
        raise

    print("Model file:", MODEL_PATH, "size=", model_size)
    print("Labels:", labels)

    try:
        net = ml.Model(
            MODEL_PATH,
            load_to_fb=model_size > (gc.mem_free() - (64 * 1024)),
        )
    except Exception as err:
        print("Model load failed:", err)
        img = sensor.snapshot()
        show(lcd, img, "LOAD FAIL", str(err), color=(255, 0, 0))
        raise

    print("Model input shape:", net.input_shape)
    print("Model output shape:", net.output_shape)

    try:
        img = sensor.snapshot()
        result = net.predict([img], callback=smoke_callback)
        print("Predict OK:", result)
    except Exception as err:
        print("Predict failed:", err)
        img = sensor.snapshot()
        show(lcd, img, "PRED FAIL", str(err), color=(255, 0, 0))
        raise

    clock = time.clock()
    while True:
        clock.tick()
        img = sensor.snapshot()
        show(lcd, img, "MODEL OK", "%.2f fps" % clock.fps(), color=(0, 255, 0))
        print("fps:", clock.fps())
        time.sleep_ms(200)


try:
    main()
except Exception as err:
    sys.print_exception(err)
    while True:
        time.sleep_ms(1000)
