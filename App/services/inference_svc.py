import timm
import asyncio
from inference.image_predictor import ImagePredictor

model_cache = {}

# ----- 사용자 업로드 Image 딥페이크 여부 분석 ------
async def predict_image(image_loc: str, model_type: str, domain_type: str):
    
    model_name = None
    dataset = None

    # 사용자 지정 Model 불러오기
    if model_type == "fast":
        if domain_type == "서양인":
            model_name = "ms_eff_gcvit_b0"; dataset = "ff++"
        else:
            model_name = "ms_eff_gcvit_b0"; dataset = "kodf"
            
    else:
        if domain_type == "서양인":
            model_name = "ms_eff_gcvit_b5"; dataset = "ff++"
        else:
            model_name = "ms_eff_gcvit_b5"; dataset = "kodf"
    
    cache_key = (model_name, dataset)
    
    if cache_key not in model_cache:
        model_cache[cache_key] = ImagePredictor(
            margin_ratio = 0.2,
            conf_thres = 0.5,
            min_face_ratio = 0.01,
            model_name = model_name,   
            dataset = dataset )
        
    predictor = model_cache[cache_key]
    
    loop = asyncio.get_running_loop()
    
    result = await loop.run_in_executor( 
        None,
        predictor.predict_img,
        f"./{image_loc}",
        0.0
        )
    
    print(f"Deepfake Probability: {result:.4f}")
    
    return result