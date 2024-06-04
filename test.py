from PIL import Image
from wmdetection.models import get_watermarks_detection_model
from wmdetection.pipelines import predictor
from wmdetection.pipelines.predictor import WatermarksPredictor


predictor: WatermarksPredictor = None


def _get_predictor():
    global predictor
    if not predictor:
        # checkpoint is automatically downloaded
        model, transforms = get_watermarks_detection_model(
            'convnext-tiny', 
            fp16=False, 
            cache_dir='/root/.cache/watermark-detection/weights'
        )
        predictor = WatermarksPredictor(model, transforms, 'cuda:0')
    return predictor


def _test_once():
    result = _get_predictor().predict_image(Image.open('images/watermark/1.jpg'))
    print('watermarked' if result else 'clean') # prints "watermarked"
    

def _test_batch():
    results = _get_predictor().run([
        'images/watermark/1.jpg',
        'images/watermark/2.jpg',
        'images/watermark/3.jpg',
        'images/watermark/4.jpg',
        'images/clean/1.jpg',
        'images/clean/2.jpg',
        'images/clean/3.jpg',
        'images/clean/4.jpg'
    ], num_workers=8, bs=8)
    for result in results:
        print('watermarked' if result else 'clean')
        
        
if __name__ == "__main__":
    # _test_once()
    _test_batch()
    print("all done")
    pass
